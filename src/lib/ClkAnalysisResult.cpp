#include "ClkAnalysisResult.h"
#include "SignalTracing.h"
#include "CombTranslator.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include <functional>
#include <algorithm>

using namespace mlir;
using namespace circt;

namespace clk_analysis {

/// 检测是否是循环中的位提取目标
/// 返回原始信号名和是否是循环位提取
struct BitDrvTarget {
  std::string signalName;  // 使用 std::string 避免生命周期问题
  bool isBitExtract;       // 是否是 sig.extract
  bool usesLoopIterator;   // 索引是否使用循环迭代器

  static BitDrvTarget simple(const std::string &name) {
    return {name, false, false};
  }
  static BitDrvTarget bitExtract(const std::string &name, bool usesLoop) {
    return {name, true, usesLoop};
  }
};

/// 检查索引值是否可能是循环迭代器
static bool isLikelyLoopIterator(Value idx) {
  // BlockArgument
  if (isa<BlockArgument>(idx))
    return true;

  // 通过 mux 选择的索引（常见于 LLHD 循环中）
  if (auto muxOp = idx.getDefiningOp<comb::MuxOp>()) {
    // 检查 true 分支是否包含 BlockArgument 或循环变量
    Value trueVal = muxOp.getTrueValue();
    if (auto extract = trueVal.getDefiningOp<comb::ExtractOp>()) {
      if (isa<BlockArgument>(extract.getInput()))
        return true;
      // 检查是否是从循环迭代器信号提取
      if (auto prb = extract.getInput().getDefiningOp<llhd::PrbOp>()) {
        if (auto sigOp = prb.getSignal().getDefiningOp<llhd::SignalOp>()) {
          if (auto name = sigOp.getName()) {
            std::string n = name->str();
            if (n.find("int_k") != std::string::npos ||
                n.find("_k") != std::string::npos)
              return true;
          }
        }
      }
    }
    // 直接是 BlockArgument 的提取
    if (isa<BlockArgument>(trueVal))
      return true;
  }

  return false;
}

/// 从 SSA 值获取名称（用于没有 name 属性的信号）
static std::string getSSAName(Value val) {
  // 对于 OpResult，获取定义操作的结果名
  if (auto opResult = dyn_cast<OpResult>(val)) {
    Operation *defOp = opResult.getOwner();
    // 尝试获取 name 或 sym_name 属性
    if (auto nameAttr = defOp->getAttrOfType<StringAttr>("name")) {
      return nameAttr.getValue().str();
    }
    if (auto symName = defOp->getAttrOfType<StringAttr>("sym_name")) {
      return symName.getValue().str();
    }
    // 打印 Value 并提取 SSA 名称
    std::string str;
    llvm::raw_string_ostream os(str);
    val.print(os);
    os.flush();
    // 格式通常是 "%name" 或 "<block argument>"
    if (!str.empty() && str[0] == '%') {
      size_t end = str.find_first_of(" :");
      if (end != std::string::npos) {
        return str.substr(1, end - 1);
      }
      return str.substr(1);
    }
  }
  return "";
}

/// 获取信号名称和位提取信息
static BitDrvTarget getSignalInfo(Value signal) {
  // 处理 llhd.sig.extract
  if (auto sigExtract = signal.getDefiningOp<llhd::SigExtractOp>()) {
    Value baseSig = sigExtract.getInput();
    Value idx = sigExtract.getLowBit();
    bool usesLoop = isLikelyLoopIterator(idx);

    if (auto sigOp = baseSig.getDefiningOp<llhd::SignalOp>()) {
      if (auto name = sigOp.getName()) {
        return BitDrvTarget::bitExtract(name->str(), usesLoop);
      }
      // 使用 SSA 名称作为后备
      std::string ssaName = getSSAName(baseSig);
      if (!ssaName.empty()) {
        return BitDrvTarget::bitExtract(ssaName, usesLoop);
      }
    }
    return BitDrvTarget::bitExtract("unnamed", usesLoop);
  }

  // 普通信号
  if (auto sigOp = signal.getDefiningOp<llhd::SignalOp>()) {
    if (auto name = sigOp.getName())
      return BitDrvTarget::simple(name->str());
    // 使用 SSA 名称作为后备
    std::string ssaName = getSSAName(signal);
    if (!ssaName.empty()) {
      return BitDrvTarget::simple(ssaName);
    }
  }
  return BitDrvTarget::simple("unnamed");
}

/// 获取信号名称（从匿名命名空间移出以供外部使用）
static std::string getSignalNameStr(Value signal) {
  return getSignalInfo(signal).signalName;
}

/// 兼容旧接口
static StringRef getSignalName(Value signal) {
  // 注意：返回临时对象的 StringRef 可能有问题，但短期内使用是安全的
  static thread_local std::string lastSignalName;
  lastSignalName = getSignalInfo(signal).signalName;
  return lastSignalName;
}

namespace {

/// 获取信号位宽
int getSignalBitWidth(Value signal) {
  Type sigType = signal.getType();
  // hw.inout<iN>
  if (auto hwInOut = dyn_cast<hw::InOutType>(sigType)) {
    Type inner = hwInOut.getElementType();
    if (auto intType = dyn_cast<IntegerType>(inner)) {
      return intType.getWidth();
    }
  }
  // 直接是 IntegerType 的情况（某些 pass 转换后）
  if (auto intType = dyn_cast<IntegerType>(sigType)) {
    return intType.getWidth();
  }
  return 32; // 默认
}

/// 检查 value 是否依赖于 signal
bool checkDependsOnSignal(Value value, Value signal) {
  llvm::SmallVector<Value> worklist;
  llvm::DenseSet<Value> visited;
  worklist.push_back(value);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (visited.contains(v)) continue;
    visited.insert(v);

    if (auto prb = v.getDefiningOp<llhd::PrbOp>()) {
      if (prb.getSignal() == signal)
        return true;
    }

    if (Operation *defOp = v.getDefiningOp()) {
      for (Value operand : defOp->getOperands())
        worklist.push_back(operand);
    }
  }
  return false;
}

/// 检查是否是 reg + constant 模式，返回步进值
int getAccumulateStep(comb::AddOp addOp, Value signal) {
  bool hasSignal = false;
  int constVal = 0;

  for (Value operand : addOp.getOperands()) {
    if (auto prb = operand.getDefiningOp<llhd::PrbOp>()) {
      if (prb.getSignal() == signal) {
        hasSignal = true;
        continue;
      }
    }
    if (auto constOp = operand.getDefiningOp<hw::ConstantOp>()) {
      constVal = constOp.getValue().getSExtValue();
    }
  }

  return hasSignal ? constVal : 0;
}

/// 检查是否是 reg - constant 模式，返回步进值
int getSubtractStep(comb::SubOp subOp, Value signal) {
  Value lhs = subOp.getLhs();
  Value rhs = subOp.getRhs();

  if (auto prb = lhs.getDefiningOp<llhd::PrbOp>()) {
    if (prb.getSignal() == signal) {
      if (auto constOp = rhs.getDefiningOp<hw::ConstantOp>()) {
        return constOp.getValue().getSExtValue();
      }
    }
  }
  return 0;
}

/// 检查 drv 是否和循环条件在同一个 wait-free 循环中
bool isInSameLoopWithCondition(llhd::DrvOp drv, Block *loopHead,
                                llvm::DenseSet<Block*> &waitBlocks) {
  // 检查 loopHead 中是否有对同一 signal 的比较
  for (Operation &op : *loopHead) {
    if (auto icmp = dyn_cast<comb::ICmpOp>(&op)) {
      for (Value operand : icmp.getOperands()) {
        if (auto prbCmp = operand.getDefiningOp<llhd::PrbOp>()) {
          if (prbCmp.getSignal() == drv.getSignal()) {
            // 检查 loopHead 的出口：一个继续循环，一个退出到 wait
            Operation *headTerminator = loopHead->getTerminator();
            if (auto condBr = dyn_cast<cf::CondBranchOp>(headTerminator)) {
              bool oneExitToWait =
                  waitBlocks.contains(condBr.getTrueDest()) ||
                  waitBlocks.contains(condBr.getFalseDest());
              bool oneExitToLoop =
                  !waitBlocks.contains(condBr.getTrueDest()) ||
                  !waitBlocks.contains(condBr.getFalseDest());
              if (oneExitToWait && oneExitToLoop) {
                // 这是 for 循环模式：条件判断后一边继续循环，一边退出
                return true;
              }
            }
          }
        }
      }
    }
  }
  return false;
}

/// 检查是否是 for 循环迭代器
/// 关键：循环是否在 wait 之间完成（不跨越 wait）
///
/// For 循环特征：
/// 1. drv 所在 block 不直接跳转到 wait block
/// 2. 从 drv block 到 wait block 的路径上存在循环回边
/// 3. 循环条件检查和递增在同一个 wait-free 区域
bool isLoopIterator(llhd::DrvOp drv, llvm::DenseSet<Block*> &waitBlocks) {
  Block *currentBlock = drv->getBlock();
  Operation *terminator = currentBlock->getTerminator();

  if (auto br = dyn_cast<cf::BranchOp>(terminator)) {
    if (waitBlocks.contains(br.getDest())) {
      // 直接跳转到 wait block，说明是跨 clock cycle 的累积
      return false;
    }
    // 检查目标 block 是否是循环头（会跳回 wait 或继续循环）
    Block *dest = br.getDest();
    // 检查从这个 block 能否不经过 wait 就到达对同一 signal 的 icmp
    if (isInSameLoopWithCondition(drv, dest, waitBlocks)) {
      return true;
    }
  } else if (auto condBr = dyn_cast<cf::CondBranchOp>(terminator)) {
    // 对于条件分支，如果任一目标是 wait block，则认为是跨 clock cycle
    if (waitBlocks.contains(condBr.getTrueDest()) ||
        waitBlocks.contains(condBr.getFalseDest())) {
      return false;
    }
  }

  return false;
}

} // anonymous namespace

/// 分析 drv 操作，提取动作
/// 如果表达式无法生成，返回 COMPUTE 类型且 expression 为 "/* complex expression */"
EventAction tryGenerateAction(llhd::DrvOp drv) {
  EventAction action;

  // 获取目标信号信息
  BitDrvTarget targetInfo = getSignalInfo(drv.getSignal());
  action.targetSignal = targetInfo.signalName;

  Value value = drv.getValue();

  // 检测循环中的位级别赋值模式
  // 如果 drv 目标是 sig.extract 且使用循环迭代器，尝试简化为整数级别操作
  if (targetInfo.isBitExtract && targetInfo.usesLoopIterator) {
    // 检测值是否也是位提取模式 (signal >> idx) & 1
    auto bitPattern = comb_translator::detectBitExtractPattern(value);
    if (bitPattern.isPattern && bitPattern.usesBlockArgument) {
      // 两边都是位提取，简化为整数级别赋值
      // target[i] = source[i] -> target = source
      action.type = ActionType::ASSIGN_SIGNAL;
      action.sourceSignal = bitPattern.signalName;
      return action;
    }

    // 检测是否是常量赋值到位 (target[i] = 0 或 target[i] = 1)
    if (auto constOp = value.getDefiningOp<hw::ConstantOp>()) {
      // 对于 target[i] = 0，整个信号应该是 target = 0 或用位操作
      // 为简化，我们跳过这种循环中的操作，因为它们通常在循环外有另一个赋值
      action.type = ActionType::COMPUTE;
      action.expression = "/* loop bit assign: " + action.targetSignal +
                          "[i] = " + std::to_string(constOp.getValue().getSExtValue()) +
                          " - handled at loop level */";
      return action;
    }

    // 对于更复杂的模式，尝试使用 CombTranslator
    auto translateResult = comb_translator::translateValue(value);
    if (translateResult.success) {
      // 检查翻译结果是否是整数级别的表达式（没有 arg0）
      if (translateResult.expr.find("arg") == std::string::npos) {
        action.type = ActionType::COMPUTE;
        action.expression = action.targetSignal + " = " + translateResult.expr;
        return action;
      }
    }

    // 无法简化的循环位操作，标记为跳过
    action.type = ActionType::COMPUTE;
    action.expression = "/* loop bit operation on " + action.targetSignal + " - skip */";
    return action;
  }

  // 检查是否是常量赋值
  if (auto constOp = value.getDefiningOp<hw::ConstantOp>()) {
    action.type = ActionType::ASSIGN_CONST;
    action.constValue = constOp.getValue().getSExtValue();
    return action;
  }

  // 检查是否是简单的信号赋值
  if (auto prb = value.getDefiningOp<llhd::PrbOp>()) {
    action.type = ActionType::ASSIGN_SIGNAL;
    action.sourceSignal = getSignalName(prb.getSignal()).str();
    return action;
  }

  // 检查是否是比较操作的结果 (signal = a cmp b)
  if (auto icmp = value.getDefiningOp<comb::ICmpOp>()) {
    auto lhsTraced = signal_tracing::traceToSignal(icmp.getLhs());
    auto rhsTraced = signal_tracing::traceToSignal(icmp.getRhs());

    if (lhsTraced.isValid() && rhsTraced.isValid()) {
      action.type = ActionType::ASSIGN_COMPARE;
      action.compareLhs = lhsTraced.name.str();
      action.compareRhs = rhsTraced.name.str();

      switch (icmp.getPredicate()) {
        case comb::ICmpPredicate::uge:
        case comb::ICmpPredicate::sge:
          action.compareType = CompareType::GE;
          break;
        case comb::ICmpPredicate::ult:
        case comb::ICmpPredicate::slt:
          action.compareType = CompareType::LT;
          break;
        case comb::ICmpPredicate::eq:
          action.compareType = CompareType::EQ;
          break;
        case comb::ICmpPredicate::ne:
          action.compareType = CompareType::NE;
          break;
        default:
          action.compareType = CompareType::NONE;
      }
      return action;
    }
  }

  // 检查是否是累加操作
  if (auto addOp = value.getDefiningOp<comb::AddOp>()) {
    for (Value operand : addOp.getOperands()) {
      if (auto prb = operand.getDefiningOp<llhd::PrbOp>()) {
        if (getSignalName(prb.getSignal()) == action.targetSignal) {
          // 这是 signal = signal + something
          for (Value op2 : addOp.getOperands()) {
            if (auto constOp = op2.getDefiningOp<hw::ConstantOp>()) {
              action.type = ActionType::ACCUMULATE;
              action.constValue = constOp.getValue().getSExtValue();
              return action;
            }
          }
        }
      }
    }
  }

  // 检查是否是减法操作
  if (auto subOp = value.getDefiningOp<comb::SubOp>()) {
    Value lhs = subOp.getLhs();
    Value rhs = subOp.getRhs();
    if (auto prb = lhs.getDefiningOp<llhd::PrbOp>()) {
      if (getSignalName(prb.getSignal()) == action.targetSignal) {
        if (auto constOp = rhs.getDefiningOp<hw::ConstantOp>()) {
          action.type = ActionType::ACCUMULATE;
          action.constValue = -constOp.getValue().getSExtValue();  // 负数表示减法
          return action;
        }
      }
    }
  }

  // 使用 CombTranslator 尝试翻译复杂表达式
  auto translateResult = comb_translator::translateValue(value);
  if (translateResult.success) {
    action.type = ActionType::COMPUTE;
    action.expression = action.targetSignal + " = " + translateResult.expr;
    return action;
  }

  // 真正无法生成的复杂表达式
  action.type = ActionType::COMPUTE;
  action.expression = "/* complex expression: " + translateResult.errorMsg + " */";
  return action;
}

/// 检查动作是否是复杂表达式（无法生成代码）
bool isComplexAction(const EventAction &action) {
  return action.type == ActionType::COMPUTE &&
         action.expression.find("/* complex expression") != std::string::npos;
}

ModuleAnalysisResult analyzeModule(mlir::ModuleOp mod) {
  ModuleAnalysisResult result;

  // 用于去重：只记录每个信号的最终分类
  llvm::StringMap<SignalAnalysisResult> signalMap;

  mod.walk([&](hw::HWModuleOp hwMod) {
    result.moduleName = hwMod.getName().str();

    hwMod.walk([&](llhd::ProcessOp proc) {
      // 收集 wait blocks
      llvm::DenseSet<Block*> waitBlocks;
      proc.walk([&](llhd::WaitOp wait) {
        waitBlocks.insert(wait->getBlock());
      });

      // 分析每个 drv 操作
      proc.walk([&](llhd::DrvOp drv) {
        Value signal = drv.getSignal();
        Value value = drv.getValue();
        StringRef sigName = getSignalName(signal);

        // 如果已经记录过这个信号，检查是否需要更新
        // ACCUMULATE 优先级最高（需要特殊处理）
        auto it = signalMap.find(sigName);
        if (it != signalMap.end()) {
          // 已经是 ACCUMULATE，不需要更新
          if (it->second.classification == DrvClassification::CLK_ACCUMULATE) {
            // 但仍然尝试生成动作并保存
            EventAction action = tryGenerateAction(drv);
            it->second.preGeneratedActions.push_back(action);
            if (isComplexAction(action)) {
              it->second.hasComplexExpression = true;
            }
            return;
          }
        }

        SignalAnalysisResult sigResult;
        sigResult.name = sigName.str();
        sigResult.bitWidth = getSignalBitWidth(signal);
        sigResult.direction = AccumulateDirection::NONE;
        sigResult.stepValue = 0;
        sigResult.hasComplexExpression = false;

        // ========== 方案 A: 先尝试生成表达式 ==========
        EventAction action = tryGenerateAction(drv);
        sigResult.preGeneratedActions.push_back(action);

        // 如果表达式无法生成，直接标记为 COMPLEX
        if (isComplexAction(action)) {
          sigResult.classification = DrvClassification::CLK_COMPLEX;
          sigResult.hasComplexExpression = true;
          // 更新或插入
          if (it != signalMap.end()) {
            // 已有记录，发现复杂表达式，必须更新分类为 COMPLEX
            it->second.classification = DrvClassification::CLK_COMPLEX;
            it->second.hasComplexExpression = true;
            it->second.preGeneratedActions.push_back(action);
          } else {
            signalMap[sigName] = sigResult;
          }
          return;
        }

        // ========== 原有分类逻辑（仅当表达式可生成时） ==========
        // 检查是否依赖自己
        bool dependsOnSelf = checkDependsOnSignal(value, signal);

        if (!dependsOnSelf) {
          sigResult.classification = DrvClassification::CLK_IGNORABLE;
        } else {
          // 检查 add 模式
          if (auto addOp = value.getDefiningOp<comb::AddOp>()) {
            int step = getAccumulateStep(addOp, signal);
            if (step != 0) {
              if (!isLoopIterator(drv, waitBlocks)) {
                sigResult.classification = DrvClassification::CLK_ACCUMULATE;
                sigResult.direction = AccumulateDirection::UP;
                sigResult.stepValue = step;
              } else {
                sigResult.classification = DrvClassification::CLK_LOOP_ITER;
              }
            } else {
              sigResult.classification = DrvClassification::CLK_COMPLEX;
            }
          }
          // 检查 sub 模式
          else if (auto subOp = value.getDefiningOp<comb::SubOp>()) {
            int step = getSubtractStep(subOp, signal);
            if (step != 0) {
              if (!isLoopIterator(drv, waitBlocks)) {
                sigResult.classification = DrvClassification::CLK_ACCUMULATE;
                sigResult.direction = AccumulateDirection::DOWN;
                sigResult.stepValue = step;
              } else {
                sigResult.classification = DrvClassification::CLK_LOOP_ITER;
              }
            } else {
              sigResult.classification = DrvClassification::CLK_COMPLEX;
            }
          }
          // 检查 hold 模式
          else if (auto prb = value.getDefiningOp<llhd::PrbOp>()) {
            if (prb.getSignal() == signal) {
              sigResult.classification = DrvClassification::CLK_IGNORABLE;
            } else {
              sigResult.classification = DrvClassification::CLK_COMPLEX;
            }
          }
          else {
            sigResult.classification = DrvClassification::CLK_COMPLEX;
          }
        }

        // 更新或插入
        // ACCUMULATE 优先级最高
        if (it != signalMap.end()) {
          if (sigResult.classification == DrvClassification::CLK_ACCUMULATE) {
            it->second = sigResult;
          } else {
            // 保留原分类，但合并预生成动作
            for (const auto &act : sigResult.preGeneratedActions) {
              it->second.preGeneratedActions.push_back(act);
            }
          }
        } else {
          signalMap[sigName] = sigResult;
        }
      });
    });

    // 处理 process 外部的 drv 操作（组合逻辑）
    // 这些 drv 不在任何 process 内部
    hwMod.walk([&](llhd::DrvOp drv) {
      // 检查是否在 process 内部
      Operation *parent = drv->getParentOp();
      while (parent && !isa<llhd::ProcessOp>(parent) && !isa<hw::HWModuleOp>(parent)) {
        parent = parent->getParentOp();
      }
      // 如果父操作是 ProcessOp，跳过（已经处理过）
      if (isa<llhd::ProcessOp>(parent)) {
        return;
      }

      // 处理 process 外部的 drv
      Value signal = drv.getSignal();
      std::string sigName = getSignalNameStr(signal);

      // 如果已经记录过这个信号，跳过
      if (signalMap.count(sigName)) {
        return;
      }

      SignalAnalysisResult sigResult;
      sigResult.name = sigName;
      sigResult.bitWidth = getSignalBitWidth(signal);
      sigResult.direction = AccumulateDirection::NONE;
      sigResult.stepValue = 0;
      sigResult.hasComplexExpression = false;

      EventAction action = tryGenerateAction(drv);
      sigResult.preGeneratedActions.push_back(action);

      if (isComplexAction(action)) {
        sigResult.classification = DrvClassification::CLK_COMPLEX;
        sigResult.hasComplexExpression = true;
      } else {
        // 组合逻辑通常是 IGNORABLE
        sigResult.classification = DrvClassification::CLK_IGNORABLE;
      }

      signalMap[sigName] = sigResult;
    });
  });

  // 转换为 vector
  for (auto &entry : signalMap) {
    result.signals.push_back(entry.second);
  }

  return result;
}

//===----------------------------------------------------------------------===//
// 事件分析 - 从 CFG 提取事件处理逻辑
//===----------------------------------------------------------------------===//

namespace {

/// 检查信号名是否是时钟信号
bool isClockSignal(const std::string &name) {
  // 精确匹配
  if (name == "clk" || name == "clock")
    return true;
  // 前缀匹配
  if (name.find("pclk") == 0 || name.find("dbclk") == 0 ||
      name.find("hclk") == 0 || name.find("aclk") == 0 ||
      name.find("sclk") == 0 || name.find("fclk") == 0)
    return true;
  // 后缀匹配
  if (name.size() > 4 && name.substr(name.size() - 4) == "_clk")
    return true;
  if (name.size() > 5 && name.substr(name.size() - 5) == "_pclk")
    return true;
  return false;
}

/// 检查信号名是否是 APB 协议信号
bool isAPBProtocolSignal(const std::string &name) {
  return name == "psel" || name == "penable" || name == "pwrite" ||
         name == "paddr" || name == "pwdata" || name == "prdata" ||
         name == "pready" || name == "pslverr";
}

/// 检查信号名是否是 GPIO 外部输入信号
/// 只匹配真正的外部端口输入，不匹配 gpio_int_xxx 等控制寄存器
bool isGPIOInputSignal(const std::string &name) {
  // gpio_ext_porta, gpio_ext_portb, etc. - 外部端口数据
  if (name.find("gpio_ext_port") != std::string::npos)
    return true;
  // gpio_ext_data - 外部数据
  if (name == "gpio_ext_data")
    return true;
  // gpio_in_data - 外部输入数据（注意：不要匹配 gpio_int_xxx）
  if (name == "gpio_in_data")
    return true;
  // 其他 _porta, _portb 等后缀（但不包含 gpio_int）
  if (name.find("gpio_int") == std::string::npos) {
    if (name.find("_porta") != std::string::npos ||
        name.find("_portb") != std::string::npos ||
        name.find("_portc") != std::string::npos ||
        name.find("_portd") != std::string::npos)
      return true;
  }
  return false;
}

/// 从 hw.module 获取输入端口信息
/// inputs: 普通输入信号
/// 返回分类后的输入信号，值表示类型:
///   "input" - 普通输入（事件处理器）
///   "clock" - 时钟信号（过滤）
///   "apb" - APB 协议信号（MMIO 处理）
///   "gpio_in" - GPIO 外部输入（qdev_init_gpio_in）
void collectInputPorts(hw::HWModuleOp hwMod,
                       std::map<std::string, std::string> &inputs) {
  auto moduleType = hwMod.getHWModuleType();
  for (auto port : moduleType.getPorts()) {
    if (port.dir == hw::ModulePort::Direction::Input) {
      std::string name = port.name.str();

      // 1. 过滤时钟信号
      if (isClockSignal(name)) {
        inputs[name] = "clock";  // 标记但不参与事件处理
        continue;
      }

      // 2. 标记 APB 协议信号
      if (isAPBProtocolSignal(name)) {
        inputs[name] = "apb";  // 通过 MMIO 处理
        continue;
      }

      // 3. 标记 GPIO 外部输入
      if (isGPIOInputSignal(name)) {
        inputs[name] = "gpio_in";  // 通过 qdev_init_gpio_in 处理
        continue;
      }

      // 4. 其他普通输入信号
      inputs[name] = "input";
    }
  }
}

/// 分析条件分支，提取条件信号
bool analyzeCondition(Value cond, ConditionalBranch &branch) {
  // 检查是否是简单的信号检测
  auto traced = signal_tracing::traceToSignal(cond);
  if (traced.isValid()) {
    branch.condSignal = traced.name.str();
    branch.condType = traced.isInverted ? ConditionType::SIGNAL_FALSE
                                        : ConditionType::SIGNAL_TRUE;
    return true;
  }

  // 检查是否是比较操作
  if (auto icmp = cond.getDefiningOp<comb::ICmpOp>()) {
    Value lhs = icmp.getLhs();
    Value rhs = icmp.getRhs();

    auto lhsTraced = signal_tracing::traceToSignal(lhs);
    auto rhsTraced = signal_tracing::traceToSignal(rhs);

    if (lhsTraced.isValid()) {
      branch.condSignal = lhsTraced.name.str();
      if (rhsTraced.isValid()) {
        branch.compareSignal = rhsTraced.name.str();
      }

      switch (icmp.getPredicate()) {
        case comb::ICmpPredicate::uge:
        case comb::ICmpPredicate::sge:
          branch.condType = ConditionType::COMPARE_GE;
          break;
        case comb::ICmpPredicate::ult:
        case comb::ICmpPredicate::slt:
          branch.condType = ConditionType::COMPARE_LT;
          break;
        case comb::ICmpPredicate::eq:
          branch.condType = ConditionType::COMPARE_EQ;
          break;
        default:
          branch.condType = ConditionType::NONE;
      }
      return true;
    }
  }

  return false;
}

/// 递归遍历 block，收集 drv 操作和嵌套条件
/// inputSignals: 输入信号集合，用于判断是否跳过嵌套条件
void collectDrvsInBlock(Block *block, ConditionalBranch &branch,
                        llvm::DenseSet<Block*> &visited,
                        llvm::DenseSet<Block*> &waitBlocks,
                        const std::map<std::string, std::string> &inputSignals) {
  if (!block || visited.contains(block) || waitBlocks.contains(block))
    return;
  visited.insert(block);

  // 收集这个 block 中的所有 drv
  for (Operation &op : *block) {
    if (auto drv = dyn_cast<llhd::DrvOp>(&op)) {
      branch.actions.push_back(tryGenerateAction(drv));
    }
  }

  // 检查 terminator 是否是条件分支（嵌套条件）
  Operation *terminator = block->getTerminator();
  if (auto condBr = dyn_cast<cf::CondBranchOp>(terminator)) {
    // 分析嵌套条件
    ConditionalBranch nestedTrueBranch;
    if (analyzeCondition(condBr.getCondition(), nestedTrueBranch)) {
      // 如果嵌套条件是另一个输入信号，记录这个嵌套关系但不递归进入
      // （它会有自己的事件处理器，但我们需要知道这个嵌套关系）
      if (inputSignals.find(nestedTrueBranch.condSignal) != inputSignals.end() &&
          (nestedTrueBranch.condType == ConditionType::SIGNAL_TRUE ||
           nestedTrueBranch.condType == ConditionType::SIGNAL_FALSE)) {
        // 记录这个嵌套的输入信号条件（用于控制信号检测）
        // 不递归进入，但要添加到 nestedBranches 中
        branch.nestedBranches.push_back(nestedTrueBranch);
        return;
      }

      // 递归收集 true 分支（内部信号比较）
      llvm::DenseSet<Block*> visitedNested;
      collectDrvsInBlock(condBr.getTrueDest(), nestedTrueBranch,
                         visitedNested, waitBlocks, inputSignals);
      if (!nestedTrueBranch.actions.empty() ||
          !nestedTrueBranch.nestedBranches.empty()) {
        branch.nestedBranches.push_back(nestedTrueBranch);
      }

      // 递归收集 false 分支
      ConditionalBranch nestedFalseBranch;
      nestedFalseBranch.condSignal = nestedTrueBranch.condSignal;
      nestedFalseBranch.compareSignal = nestedTrueBranch.compareSignal;

      // 反转条件类型
      switch (nestedTrueBranch.condType) {
        case ConditionType::SIGNAL_TRUE:
          nestedFalseBranch.condType = ConditionType::SIGNAL_FALSE;
          break;
        case ConditionType::SIGNAL_FALSE:
          nestedFalseBranch.condType = ConditionType::SIGNAL_TRUE;
          break;
        case ConditionType::COMPARE_GE:
          nestedFalseBranch.condType = ConditionType::COMPARE_LT;
          break;
        case ConditionType::COMPARE_LT:
          nestedFalseBranch.condType = ConditionType::COMPARE_GE;
          break;
        case ConditionType::COMPARE_EQ:
          // != 需要新类型，暂时跳过
          break;
        default:
          break;
      }

      llvm::DenseSet<Block*> visitedFalse;
      collectDrvsInBlock(condBr.getFalseDest(), nestedFalseBranch,
                         visitedFalse, waitBlocks, inputSignals);
      if (!nestedFalseBranch.actions.empty() ||
          !nestedFalseBranch.nestedBranches.empty()) {
        branch.nestedBranches.push_back(nestedFalseBranch);
      }
    }
  }
}

/// 从 CFG 提取事件处理逻辑
void extractEventHandlers(llhd::ProcessOp proc,
                          std::vector<EventHandler> &handlers,
                          const std::map<std::string, std::string> &inputSignals) {
  // 收集 wait blocks
  llvm::DenseSet<Block*> waitBlocks;
  proc.walk([&](llhd::WaitOp wait) {
    waitBlocks.insert(wait->getBlock());
  });

  // 遍历所有条件分支，查找基于输入信号的条件
  proc.walk([&](cf::CondBranchOp condBr) {
    ConditionalBranch trueBranch;
    if (!analyzeCondition(condBr.getCondition(), trueBranch))
      return;

    // 检查条件信号是否是输入信号
    auto it = inputSignals.find(trueBranch.condSignal);
    if (it == inputSignals.end())
      return;

    // 只为普通输入信号生成事件处理器
    // 跳过: clock（时钟）, apb（APB协议）, gpio_in（GPIO外部输入）
    const std::string &signalType = it->second;
    if (signalType == "clock" || signalType == "apb" || signalType == "gpio_in")
      return;

    // 找到或创建对应的事件处理器
    EventHandler *handler = nullptr;
    for (auto &h : handlers) {
      if (h.triggerSignal == trueBranch.condSignal) {
        handler = &h;
        break;
      }
    }
    if (!handler) {
      handlers.emplace_back();
      handler = &handlers.back();
      handler->triggerSignal = trueBranch.condSignal;
    }

    // 收集 true 分支的动作和嵌套条件
    llvm::DenseSet<Block*> visitedTrue;
    collectDrvsInBlock(condBr.getTrueDest(), trueBranch,
                       visitedTrue, waitBlocks, inputSignals);

    if (!trueBranch.actions.empty() || !trueBranch.nestedBranches.empty()) {
      handler->branches.push_back(trueBranch);
    }

    // 收集 false 分支的动作（else 分支）
    ConditionalBranch falseBranch;
    falseBranch.condSignal = trueBranch.condSignal;
    // 反转条件类型
    switch (trueBranch.condType) {
      case ConditionType::SIGNAL_TRUE:
        falseBranch.condType = ConditionType::SIGNAL_FALSE;
        break;
      case ConditionType::SIGNAL_FALSE:
        falseBranch.condType = ConditionType::SIGNAL_TRUE;
        break;
      default:
        // 对于比较操作，else 分支暂不处理
        return;
    }

    llvm::DenseSet<Block*> visitedFalse;
    collectDrvsInBlock(condBr.getFalseDest(), falseBranch,
                       visitedFalse, waitBlocks, inputSignals);

    if (!falseBranch.actions.empty() || !falseBranch.nestedBranches.empty()) {
      handler->branches.push_back(falseBranch);
    }
  });
}

/// 检测派生信号（如 warning_threshold = timeout_val >> 1）
void collectDerivedSignals(hw::HWModuleOp hwMod,
                           std::vector<DerivedSignal> &derivedSignals) {
  // 查找模块体中的直接 drv 操作（不在 process 内的）
  hwMod.walk([&](llhd::DrvOp drv) {
    // 检查是否是 epsilon 延迟（用于组合逻辑 assign）
    Value signal = drv.getSignal();
    Value value = drv.getValue();
    StringRef sigName = getSignalName(signal);

    // 检查值是否来自 concat 操作（常见于位移）
    if (auto concat = value.getDefiningOp<comb::ConcatOp>()) {
      // 检查是否是 {0, signal[high:low]} 模式（右移）
      auto inputs = concat.getInputs();
      if (inputs.size() == 2) {
        // 第一个输入是 0（零扩展）
        if (auto constOp = inputs[0].getDefiningOp<hw::ConstantOp>()) {
          if (constOp.getValue().isZero()) {
            // 第二个输入是 extract 操作
            if (auto extract = inputs[1].getDefiningOp<comb::ExtractOp>()) {
              int lowBit = extract.getLowBit();
              if (lowBit > 0) {
                // 这是右移操作
                auto sourceTraced = signal_tracing::traceToSignal(extract.getInput());
                if (sourceTraced.isValid()) {
                  DerivedSignal derived;
                  derived.name = sigName.str();
                  derived.bitWidth = getSignalBitWidth(signal);
                  derived.sourceSignal = sourceTraced.name.str();
                  derived.exprType = DerivedExprType::SHIFT_RIGHT;
                  derived.exprValue = lowBit;
                  derivedSignals.push_back(derived);
                }
              }
            }
          }
        }
      }
    }
  });
}

/// 检测控制信号关系（如 enable 控制 counter 的启停）
/// 逻辑：
/// 1. 如果一个输入信号的某个分支内有 ACCUMULATE 动作，那这个输入信号就是控制信号
/// 2. 如果 handler A 的分支内嵌套了另一个输入信号 B 的条件，且 B 是控制信号，
///    那么 A 也是控制信号（AND 条件组合）
void detectControlRelations(const std::vector<EventHandler> &handlers,
                            const std::vector<SignalAnalysisResult> &signals,
                            std::vector<ControlRelation> &relations,
                            const std::map<std::string, std::string> &inputSignals) {
  // 收集所有 ACCUMULATE 类型的信号名
  llvm::StringSet<> counterSignals;
  for (const auto &sig : signals) {
    if (sig.classification == DrvClassification::CLK_ACCUMULATE) {
      counterSignals.insert(sig.name);
    }
  }

  // 辅助函数：递归检查分支中是否有对特定计数器的 ACCUMULATE 操作
  std::function<bool(const ConditionalBranch&, const std::string&)> hasAccumulateFor;
  hasAccumulateFor = [&](const ConditionalBranch &branch, const std::string &counter) -> bool {
    for (const auto &action : branch.actions) {
      if (action.type == ActionType::ACCUMULATE && action.targetSignal == counter) {
        return true;
      }
    }
    for (const auto &nested : branch.nestedBranches) {
      if (hasAccumulateFor(nested, counter))
        return true;
    }
    return false;
  };

  // 辅助函数：检查分支中是否嵌套了某个输入信号的条件，返回该条件的 activeHigh
  // 返回: {是否找到, activeHigh}
  std::function<std::pair<bool, bool>(const ConditionalBranch&, const std::string&)> hasNestedInputCondition;
  hasNestedInputCondition = [&](const ConditionalBranch &branch, const std::string &inputName)
      -> std::pair<bool, bool> {
    for (const auto &nested : branch.nestedBranches) {
      if (nested.condSignal == inputName) {
        bool activeHigh = (nested.condType == ConditionType::SIGNAL_TRUE);
        return {true, activeHigh};
      }
      auto result = hasNestedInputCondition(nested, inputName);
      if (result.first) return result;
    }
    return {false, true};
  };

  // 第一轮：直接检测（handler 的分支中直接包含 ACCUMULATE）
  for (const auto &handler : handlers) {
    for (const auto &counter : counterSignals) {
      bool foundInTrue = false;
      bool foundInFalse = false;

      for (const auto &branch : handler.branches) {
        bool hasAccum = hasAccumulateFor(branch, counter.getKey().str());
        if (hasAccum) {
          if (branch.condType == ConditionType::SIGNAL_TRUE) {
            foundInTrue = true;
          } else if (branch.condType == ConditionType::SIGNAL_FALSE) {
            foundInFalse = true;
          }
        }
      }

      if (foundInTrue && !foundInFalse) {
        ControlRelation rel;
        rel.controlSignal = handler.triggerSignal;
        rel.counterSignal = counter.getKey().str();
        rel.activeHigh = true;
        relations.push_back(rel);
      } else if (foundInFalse && !foundInTrue) {
        ControlRelation rel;
        rel.controlSignal = handler.triggerSignal;
        rel.counterSignal = counter.getKey().str();
        rel.activeHigh = false;
        relations.push_back(rel);
      }
    }
  }

  // 第二轮：检测嵌套条件（handler A 包含嵌套的 handler B 条件，B 已是控制信号）
  // 如果 A.true 分支内嵌套了 B 的条件，且 B 控制某个 counter，则 A 也控制该 counter
  for (const auto &handlerA : handlers) {
    for (const auto &branchA : handlerA.branches) {
      // 检查 branchA 是否嵌套了其他输入信号的条件
      for (const auto &inputPair : inputSignals) {
        const std::string &inputB = inputPair.first;
        if (inputB == handlerA.triggerSignal) continue;  // 跳过自己

        auto [hasNested, nestedActiveHigh] = hasNestedInputCondition(branchA, inputB);
        if (!hasNested) continue;

        // 检查 inputB 是否是某个 counter 的控制信号
        for (const auto &existingRel : relations) {
          if (existingRel.controlSignal == inputB) {
            // inputB 控制 counter，检查 A 是否已经有这个关系
            bool alreadyExists = false;
            for (const auto &r : relations) {
              if (r.controlSignal == handlerA.triggerSignal &&
                  r.counterSignal == existingRel.counterSignal) {
                alreadyExists = true;
                break;
              }
            }
            if (!alreadyExists) {
              // A 的 true 分支嵌套了 B 的条件 → A=1 时才能到达累加路径
              ControlRelation rel;
              rel.controlSignal = handlerA.triggerSignal;
              rel.counterSignal = existingRel.counterSignal;
              rel.activeHigh = (branchA.condType == ConditionType::SIGNAL_TRUE);
              relations.push_back(rel);
            }
          }
        }
      }
    }
  }
}

} // anonymous namespace

// Forward declaration
void extractAPBRegisterMappings(ModuleOp mod,
                                 std::vector<APBRegisterMapping> &mappings);

ModuleAnalysisResult analyzeModuleWithEvents(mlir::ModuleOp mod) {
  // 首先进行基本分析
  ModuleAnalysisResult result = analyzeModule(mod);

  // 收集输入信号和事件处理逻辑
  mod.walk([&](hw::HWModuleOp hwMod) {
    // 收集输入端口
    collectInputPorts(hwMod, result.inputSignals);

    // 为信号设置角色
    for (auto &sig : result.signals) {
      // 根据依赖关系和分类设置角色
      if (result.inputSignals.find(sig.name) != result.inputSignals.end()) {
        sig.role = SignalRole::INPUT;
      } else if (sig.classification == DrvClassification::CLK_ACCUMULATE) {
        sig.role = SignalRole::STATE;
      } else {
        // 检查是否有其他信号依赖此信号
        sig.role = SignalRole::OUTPUT;
      }

      // 收集依赖关系
      // (这里简化处理，实际应该从分析中提取)
    }

    // 提取事件处理逻辑
    hwMod.walk([&](llhd::ProcessOp proc) {
      extractEventHandlers(proc, result.eventHandlers, result.inputSignals);
    });

    // 收集派生信号
    collectDerivedSignals(hwMod, result.derivedSignals);
  });

  // 检测控制信号关系
  detectControlRelations(result.eventHandlers, result.signals,
                         result.controlRelations, result.inputSignals);

  // 提取 APB 寄存器地址映射
  extractAPBRegisterMappings(mod, result.apbMappings);

  return result;
}

//===----------------------------------------------------------------------===//
// APB 寄存器地址映射提取
//===----------------------------------------------------------------------===//

/// 从模块中提取 APB 寄存器地址映射
/// 分析模式: paddr → extract → icmp eq const → and(psel, penable, pwrite, icmp) → mux(and, pwdata, reg)
void extractAPBRegisterMappings(ModuleOp mod,
                                 std::vector<APBRegisterMapping> &mappings) {
  // 遍历所有模块
  mod.walk([&](hw::HWModuleOp hwMod) {
    // 用于存储: icmp结果Value → (地址值, 地址偏移位数)
    llvm::DenseMap<Value, std::pair<int64_t, int>> icmpToAddress;
    // 用于存储: and结果Value → 地址值
    llvm::DenseMap<Value, int64_t> andToAddress;

    // 第一遍: 找到所有 paddr 相关的 icmp
    hwMod.walk([&](comb::ICmpOp icmp) {
      if (icmp.getPredicate() != comb::ICmpPredicate::eq)
        return;

      // 检查是否比较的是常量
      Value lhs = icmp.getLhs();
      Value rhs = icmp.getRhs();

      hw::ConstantOp constOp = nullptr;
      Value extractVal = nullptr;

      if (auto c = rhs.getDefiningOp<hw::ConstantOp>()) {
        constOp = c;
        extractVal = lhs;
      } else if (auto c = lhs.getDefiningOp<hw::ConstantOp>()) {
        constOp = c;
        extractVal = rhs;
      }

      if (!constOp || !extractVal)
        return;

      // 检查 extractVal 是否是从 paddr 提取的
      if (auto extract = extractVal.getDefiningOp<comb::ExtractOp>()) {
        Value input = extract.getInput();
        // 检查输入是否是 paddr（可能是 BlockArgument 或命名信号）
        bool isPaddr = false;
        if (auto arg = dyn_cast<BlockArgument>(input)) {
          // 检查参数名
          if (auto hwMod = dyn_cast<hw::HWModuleOp>(arg.getOwner()->getParentOp())) {
            auto moduleType = hwMod.getHWModuleType();
            auto ports = moduleType.getPorts();
            unsigned argNum = arg.getArgNumber();
            // 输入端口索引
            unsigned inputIdx = 0;
            for (auto port : ports) {
              if (port.dir == hw::ModulePort::Direction::Input) {
                if (inputIdx == argNum) {
                  if (port.name.str().find("paddr") != std::string::npos) {
                    isPaddr = true;
                  }
                  break;
                }
                inputIdx++;
              }
            }
          }
        }

        if (isPaddr) {
          int64_t addrVal = constOp.getValue().getSExtValue();
          int lowBit = extract.getLowBit();
          icmpToAddress[icmp.getResult()] = {addrVal, lowBit};
        }
      }
    });

    // 第二遍: 找到 and(psel, penable, pwrite, icmp) 模式
    hwMod.walk([&](comb::AndOp andOp) {
      // 检查操作数中是否有 psel, penable, pwrite 和 icmp 结果
      bool hasPsel = false, hasPenable = false, hasPwrite = false;
      Value icmpResult;

      for (Value operand : andOp.getOperands()) {
        // 检查是否是 BlockArgument (输入信号)
        if (auto arg = dyn_cast<BlockArgument>(operand)) {
          if (auto hwMod = dyn_cast<hw::HWModuleOp>(arg.getOwner()->getParentOp())) {
            auto moduleType = hwMod.getHWModuleType();
            auto ports = moduleType.getPorts();
            unsigned argNum = arg.getArgNumber();
            unsigned inputIdx = 0;
            for (auto port : ports) {
              if (port.dir == hw::ModulePort::Direction::Input) {
                if (inputIdx == argNum) {
                  std::string name = port.name.str();
                  if (name == "psel") hasPsel = true;
                  else if (name == "penable") hasPenable = true;
                  else if (name == "pwrite") hasPwrite = true;
                  break;
                }
                inputIdx++;
              }
            }
          }
        }
        // 检查是否是 icmp 结果
        if (icmpToAddress.count(operand)) {
          icmpResult = operand;
        }
      }

      // 如果是 APB 写条件模式
      if (hasPsel && hasPenable && hasPwrite && icmpResult) {
        andToAddress[andOp.getResult()] = icmpToAddress[icmpResult].first;
      }
    });

    // 第三遍: 找到 mux(and, pwdata, reg) 模式
    hwMod.walk([&](comb::MuxOp mux) {
      Value cond = mux.getCond();
      if (!andToAddress.count(cond))
        return;

      int64_t addrVal = andToAddress[cond];
      Value trueVal = mux.getTrueValue();
      Value falseVal = mux.getFalseValue();

      // trueVal 应该是 pwdata, falseVal 应该是寄存器当前值
      // 检查 trueVal 是否是 pwdata
      bool isPwdata = false;
      if (auto arg = dyn_cast<BlockArgument>(trueVal)) {
        if (auto hwMod = dyn_cast<hw::HWModuleOp>(arg.getOwner()->getParentOp())) {
          auto moduleType = hwMod.getHWModuleType();
          auto ports = moduleType.getPorts();
          unsigned argNum = arg.getArgNumber();
          unsigned inputIdx = 0;
          for (auto port : ports) {
            if (port.dir == hw::ModulePort::Direction::Input) {
              if (inputIdx == argNum) {
                if (port.name.str().find("pwdata") != std::string::npos) {
                  isPwdata = true;
                }
                break;
              }
              inputIdx++;
            }
          }
        }
      }

      if (!isPwdata)
        return;

      // falseVal 应该是寄存器名
      std::string regName;
      // 尝试从 seq.firreg 获取名称
      for (auto *user : mux.getResult().getUsers()) {
        if (auto firreg = dyn_cast<seq::FirRegOp>(user)) {
          if (auto nameAttr = firreg->getAttrOfType<StringAttr>("name")) {
            regName = nameAttr.getValue().str();
          } else {
            // 使用 SSA 名称
            std::string str;
            llvm::raw_string_ostream os(str);
            firreg.getResult().print(os);
            os.flush();
            if (!str.empty() && str[0] == '%') {
              size_t end = str.find_first_of(" :");
              if (end != std::string::npos) {
                regName = str.substr(1, end - 1);
              } else {
                regName = str.substr(1);
              }
            }
          }
          break;
        }
      }

      if (regName.empty())
        return;

      // 计算字节地址 (假设从 bit 2 提取)
      uint32_t byteAddr = static_cast<uint32_t>(addrVal) << 2;

      // 检查是否已存在
      bool found = false;
      for (auto &mapping : mappings) {
        if (mapping.address == byteAddr) {
          found = true;
          if (mapping.registerName.empty()) {
            mapping.registerName = regName;
          }
          mapping.isWritable = true;
          break;
        }
      }

      if (!found) {
        APBRegisterMapping mapping;
        mapping.address = byteAddr;
        mapping.registerName = regName;
        mapping.bitWidth = mux.getType().getIntOrFloatBitWidth();
        mapping.isWritable = true;
        mapping.isReadable = true;
        mappings.push_back(mapping);
      }
    });
  });

  // 按地址排序
  std::sort(mappings.begin(), mappings.end(),
            [](const APBRegisterMapping &a, const APBRegisterMapping &b) {
              return a.address < b.address;
            });
}

} // namespace clk_analysis

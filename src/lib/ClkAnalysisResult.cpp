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
#include <functional>

using namespace mlir;
using namespace circt;

namespace clk_analysis {

/// 获取信号名称（从匿名命名空间移出以供外部使用）
static StringRef getSignalName(Value signal) {
  if (auto sigOp = signal.getDefiningOp<llhd::SignalOp>()) {
    if (auto name = sigOp.getName())
      return *name;
  }
  return "unnamed";
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

/// 检查是否是 for 循环迭代器
bool isLoopIterator(llhd::DrvOp drv, llvm::DenseSet<Block*> &waitBlocks) {
  Block *currentBlock = drv->getBlock();
  Operation *terminator = currentBlock->getTerminator();

  if (auto br = dyn_cast<cf::BranchOp>(terminator)) {
    if (waitBlocks.contains(br.getDest())) {
      return false;  // 直接跳转到 wait block，是跨 clock cycle 的
    }
  } else if (auto condBr = dyn_cast<cf::CondBranchOp>(terminator)) {
    if (waitBlocks.contains(condBr.getTrueDest()) ||
        waitBlocks.contains(condBr.getFalseDest())) {
      return false;
    }
  }

  return false;  // 保守处理
}

} // anonymous namespace

/// 分析 drv 操作，提取动作
/// 如果表达式无法生成，返回 COMPUTE 类型且 expression 为 "/* complex expression */"
EventAction tryGenerateAction(llhd::DrvOp drv) {
  EventAction action;
  action.targetSignal = getSignalName(drv.getSignal()).str();

  Value value = drv.getValue();

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

/// 从 hw.module 获取输入端口信息
void collectInputPorts(hw::HWModuleOp hwMod,
                       std::map<std::string, std::string> &inputs) {
  auto moduleType = hwMod.getHWModuleType();
  for (auto port : moduleType.getPorts()) {
    if (port.dir == hw::ModulePort::Direction::Input) {
      std::string name = port.name.str();
      // 跳过时钟和复位
      if (name == "clk" || name == "rst_n" || name == "reset")
        continue;
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
    if (inputSignals.find(trueBranch.condSignal) == inputSignals.end())
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

  return result;
}

} // namespace clk_analysis

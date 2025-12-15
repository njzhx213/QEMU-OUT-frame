#include <memory>
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"

// 信号追溯工具
#include "SignalTracing.h"

// 生成 pass 声明/定义
#define GEN_PASS_DECL_DFFDEMO
#define GEN_PASS_DEF_DFFDEMO
#define GEN_PASS_DECL_CLKDEPENDENCYANALYSIS
#define GEN_PASS_DEF_CLKDEPENDENCYANALYSIS
#include "Passes.h.inc"

using namespace mlir;
using namespace circt;

namespace {

//===----------------------------------------------------------------------===//
// DffDemo Pass (原有的)
//===----------------------------------------------------------------------===//

struct DffDemoPass : public ::impl::DffDemoBase<DffDemoPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    llvm::outs() << "[pass-ran] dff-demo\n";

    int numConst = 0;
    mod.walk([&](circt::hw::ConstantOp cst) { ++numConst; });
    llvm::outs() << "[summary] constOps=" << numConst << "\n";
  }
};

//===----------------------------------------------------------------------===//
// ClkDependencyAnalysis Pass (新增)
//===----------------------------------------------------------------------===//

/// 寄存器写操作的分类
enum class DrvClassification {
  CLK_IGNORABLE,   // 可忽略时钟，转换为事件驱动
  CLK_ACCUMULATE,  // 累计型，需要 icount/ptimer
  CLK_LOOP_ITER,   // for 循环迭代器，组合逻辑
  CLK_COMPLEX,     // 复杂依赖，需要进一步分析
};

/// COMPLEX 的细分类型（按优先级：Clock > Protocol > Data）
enum class ComplexSubType {
  NONE,            // 不是 COMPLEX
  CLOCK_RELATED,   // 依赖时钟信号 → 需要时钟模拟
  PROTOCOL,        // 依赖协议信号 → MMIO 回调
  DATA_ONLY,       // 其他情况 → 即时计算
};

llvm::StringRef classificationToString(DrvClassification c) {
  switch (c) {
    case DrvClassification::CLK_IGNORABLE:  return "CLK_IGNORABLE";
    case DrvClassification::CLK_ACCUMULATE: return "CLK_ACCUMULATE";
    case DrvClassification::CLK_LOOP_ITER:  return "CLK_LOOP_ITER";
    case DrvClassification::CLK_COMPLEX:    return "CLK_COMPLEX";
  }
  return "UNKNOWN";
}

struct ClkDependencyAnalysisPass
    : public ::impl::ClkDependencyAnalysisBase<ClkDependencyAnalysisPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    llvm::outs() << "========================================\n";
    llvm::outs() << "CLK Dependency Analysis for QEMU Conversion\n";
    llvm::outs() << "========================================\n\n";

    // 统计
    int totalProcesses = 0;
    int totalDrvs = 0;
    int ignorable = 0, accumulate = 0, loopIter = 0, complex = 0;
    // COMPLEX 细分统计
    int complexDataOnly = 0, complexProtocol = 0, complexClockRelated = 0;

    // 遍历所有 hw.module
    mod.walk([&](hw::HWModuleOp hwMod) {
      llvm::outs() << "Module: @" << hwMod.getName() << "\n";
      llvm::outs() << "----------------------------------------\n";

      // 遍历所有 llhd.process
      hwMod.walk([&](llhd::ProcessOp proc) {
        ++totalProcesses;
        llvm::outs() << "\n  [Process] at " << proc.getLoc() << "\n";

        // 收集 wait 的 block 信息
        llvm::DenseSet<Block*> waitBlocks;
        proc.walk([&](llhd::WaitOp wait) {
          waitBlocks.insert(wait->getBlock());
          llvm::outs() << "    [wait] observes "
                       << wait.getObserved().size() << " signals\n";
        });

        // 分析每个 drv 操作
        proc.walk([&](llhd::DrvOp drv) {
          ++totalDrvs;
          Value signal = drv.getSignal();
          Value value = drv.getValue();

          // 获取信号名称
          StringRef sigName = getSignalName(signal);

          llvm::outs() << "    [drv] %" << sigName << " <- ";

          // 分析并分类
          ComplexSubType complexSub = ComplexSubType::NONE;
          DrvClassification cls = classifyDrv(drv, signal, value, proc, waitBlocks, complexSub);

          llvm::outs() << classificationToString(cls) << "\n";

          // 统计
          switch (cls) {
            case DrvClassification::CLK_IGNORABLE:  ++ignorable; break;
            case DrvClassification::CLK_ACCUMULATE: ++accumulate; break;
            case DrvClassification::CLK_LOOP_ITER:  ++loopIter; break;
            case DrvClassification::CLK_COMPLEX:
              ++complex;
              // 细分统计
              switch (complexSub) {
                case ComplexSubType::CLOCK_RELATED:  ++complexClockRelated; break;
                case ComplexSubType::PROTOCOL:       ++complexProtocol; break;
                case ComplexSubType::DATA_ONLY:      ++complexDataOnly; break;
                default: break;
              }
              break;
          }
        });
      });
      llvm::outs() << "\n";
    });

    // 打印总结
    llvm::outs() << "========================================\n";
    llvm::outs() << "Summary:\n";
    llvm::outs() << "  Total processes: " << totalProcesses << "\n";
    llvm::outs() << "  Total drv ops:   " << totalDrvs << "\n";
    llvm::outs() << "  - CLK_IGNORABLE:  " << ignorable
                 << " (can be event-driven)\n";
    llvm::outs() << "  - CLK_ACCUMULATE: " << accumulate
                 << " (need icount/ptimer)\n";
    llvm::outs() << "  - CLK_LOOP_ITER:  " << loopIter
                 << " (for-loop, combinational)\n";
    llvm::outs() << "  - CLK_COMPLEX:    " << complex << "\n";
    if (complex > 0) {
      llvm::outs() << "      * Clock related:  " << complexClockRelated
                   << " (need clock simulation)\n";
      llvm::outs() << "      * Protocol:       " << complexProtocol
                   << " (MMIO callback)\n";
      llvm::outs() << "      * Data only:      " << complexDataOnly
                   << " (immediate compute)\n";
    }
    llvm::outs() << "========================================\n";
  }

private:
  /// 获取信号名称
  StringRef getSignalName(Value signal) {
    if (auto sigOp = signal.getDefiningOp<llhd::SignalOp>()) {
      if (auto name = sigOp.getName())
        return *name;
    }
    return "unnamed";
  }

  /// 分类一个 drv 操作
  DrvClassification classifyDrv(llhd::DrvOp drv, Value signal, Value value,
                                 llhd::ProcessOp proc,
                                 llvm::DenseSet<Block*> &waitBlocks,
                                 ComplexSubType &complexSub) {
    complexSub = ComplexSubType::NONE;
    // Step 1: 检查 value 是否依赖于 signal 自己
    bool dependsOnSelf = checkDependsOnSignal(value, signal);

    if (!dependsOnSelf) {
      // 不依赖自己 → 覆盖型，可忽略时钟
      llvm::outs() << "(overwrite) ";
      return DrvClassification::CLK_IGNORABLE;
    }

    // Step 2: 依赖自己，检查是什么模式

    // 检查是否是 add/sub 常量模式
    if (auto addOp = value.getDefiningOp<comb::AddOp>()) {
      if (isAccumulatePattern(addOp, signal)) {
        // Step 3: 检查是否是 for 循环迭代器
        if (isLoopIterator(drv, addOp, waitBlocks)) {
          llvm::outs() << "(for-loop iter) ";
          return DrvClassification::CLK_LOOP_ITER;
        }
        llvm::outs() << "(accumulate: +const) ";
        return DrvClassification::CLK_ACCUMULATE;
      }
    }

    if (auto subOp = value.getDefiningOp<comb::SubOp>()) {
      if (isSubtractPattern(subOp, signal)) {
        if (isLoopIterator(drv, subOp, waitBlocks)) {
          llvm::outs() << "(for-loop iter) ";
          return DrvClassification::CLK_LOOP_ITER;
        }
        llvm::outs() << "(accumulate: -const) ";
        return DrvClassification::CLK_ACCUMULATE;
      }
    }

    // Step 4: 检查是否是保持型 (reg <= reg)
    if (auto prb = value.getDefiningOp<llhd::PrbOp>()) {
      if (prb.getSignal() == signal) {
        llvm::outs() << "(hold) ";
        return DrvClassification::CLK_IGNORABLE;
      }
    }

    // 其他情况：复杂依赖 - 使用信号追溯分析
    llvm::outs() << "(complex";

    // 分析依赖了哪些信号
    auto deps = signal_tracing::getAllSignalDependencies(value);

    // 统计各类信号
    bool hasData = false, hasProtocol = false, hasClock = false;
    bool hasReset = false, hasEnable = false, hasOther = false;

    if (!deps.empty()) {
      llvm::outs() << ": depends on ";
      bool first = true;
      for (auto &dep : deps) {
        if (!first) llvm::outs() << ", ";
        first = false;

        // 分类信号类型
        auto sigType = signal_tracing::classifySignalByName(dep.name);
        llvm::outs() << dep.name;
        if (dep.isInverted) llvm::outs() << "(inv)";
        llvm::outs() << "[" << signal_tracing::signalTypeToString(sigType) << "]";

        // 统计
        switch (sigType) {
          case signal_tracing::SignalType::Data:     hasData = true; break;
          case signal_tracing::SignalType::Protocol: hasProtocol = true; break;
          case signal_tracing::SignalType::Clock:    hasClock = true; break;
          case signal_tracing::SignalType::Reset:    hasReset = true; break;
          case signal_tracing::SignalType::Enable:   hasEnable = true; break;
          default:                                   hasOther = true; break;
        }
      }
    }

    // 确定 COMPLEX 细分类型（优先级：Clock > Protocol > Data）
    if (hasClock) {
      complexSub = ComplexSubType::CLOCK_RELATED;
    } else if (hasProtocol) {
      complexSub = ComplexSubType::PROTOCOL;
    } else {
      complexSub = ComplexSubType::DATA_ONLY;  // 其他所有情况都可以即时计算
    }

    llvm::outs() << ") ";
    return DrvClassification::CLK_COMPLEX;
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

      // 检查是否是 prb 同一个 signal
      if (auto prb = v.getDefiningOp<llhd::PrbOp>()) {
        if (prb.getSignal() == signal)
          return true;
      }

      // 继续追溯操作数
      if (Operation *defOp = v.getDefiningOp()) {
        for (Value operand : defOp->getOperands())
          worklist.push_back(operand);
      }
    }
    return false;
  }

  /// 检查是否是 reg + constant 模式
  bool isAccumulatePattern(comb::AddOp addOp, Value signal) {
    bool hasSignal = false, hasConst = false;

    for (Value operand : addOp.getOperands()) {
      // 检查是否是 prb(signal)
      if (auto prb = operand.getDefiningOp<llhd::PrbOp>()) {
        if (prb.getSignal() == signal) {
          hasSignal = true;
          continue;
        }
      }
      // 检查是否是常量
      if (operand.getDefiningOp<hw::ConstantOp>()) {
        hasConst = true;
      }
    }

    return hasSignal && hasConst;
  }

  /// 检查是否是 reg - constant 模式
  bool isSubtractPattern(comb::SubOp subOp, Value signal) {
    Value lhs = subOp.getLhs();
    Value rhs = subOp.getRhs();

    // lhs 应该是 prb(signal)
    if (auto prb = lhs.getDefiningOp<llhd::PrbOp>()) {
      if (prb.getSignal() == signal) {
        // rhs 应该是常量
        if (rhs.getDefiningOp<hw::ConstantOp>()) {
          return true;
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
  bool isLoopIterator(llhd::DrvOp drv, Operation *addOrSubOp,
                      llvm::DenseSet<Block*> &waitBlocks) {
    Block *currentBlock = drv->getBlock();

    // 检查 drv 所在 block 的终结符是否直接跳转到 wait block
    // 如果直接跳转到 wait block，说明这是跨 clock cycle 的操作
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
};

} // namespace

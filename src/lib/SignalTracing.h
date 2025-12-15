#ifndef SIGNAL_TRACING_H
#define SIGNAL_TRACING_H

#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace signal_tracing {

//===----------------------------------------------------------------------===//
// 信号追溯结果
//===----------------------------------------------------------------------===//

struct TracedSignal {
  mlir::Value signal;           // 原始 llhd.sig
  llvm::StringRef name;         // 信号名
  bool isInverted;              // 是否被取反

  bool isValid() const { return !name.empty(); }
};

//===----------------------------------------------------------------------===//
// 信号追溯函数
//===----------------------------------------------------------------------===//

/// 追踪一个 Value 的源头，穿透 XOR(取反)、PRB(探测) 等操作
inline TracedSignal traceToSignal(mlir::Value val) {
  TracedSignal result;
  result.isInverted = false;

  // 1. 穿透取反逻辑 (XOR true)
  while (auto xorOp = val.getDefiningOp<circt::comb::XorOp>()) {
    if (xorOp.getNumOperands() != 2) break;

    mlir::Value lhs = xorOp.getOperand(0);
    mlir::Value rhs = xorOp.getOperand(1);

    // 检查 rhs 是否为常数 1 (true)
    if (auto constOp = rhs.getDefiningOp<circt::hw::ConstantOp>()) {
      if (constOp.getValue().isAllOnes()) {
        val = lhs;
        result.isInverted = !result.isInverted;
        continue;
      }
    }
    // 检查 lhs 是否为常数 1 (true)
    if (auto constOp = lhs.getDefiningOp<circt::hw::ConstantOp>()) {
      if (constOp.getValue().isAllOnes()) {
        val = rhs;
        result.isInverted = !result.isInverted;
        continue;
      }
    }
    break;
  }

  // 2. 穿透 PRB (探测)
  if (auto prbOp = val.getDefiningOp<circt::llhd::PrbOp>()) {
    mlir::Value sig = prbOp.getSignal();
    result.signal = sig;

    // 3. 获取信号名
    if (mlir::Operation* defOp = sig.getDefiningOp()) {
      if (defOp->getName().getStringRef() == "llhd.sig") {
        if (auto nameAttr = defOp->getAttrOfType<mlir::StringAttr>("name")) {
          result.name = nameAttr.getValue();
        }
      }
    }
  }

  return result;
}

/// 递归追溯 Value 的所有信号依赖
inline void traceAllDependencies(mlir::Value val,
                                  llvm::SmallVectorImpl<TracedSignal> &deps,
                                  llvm::SmallPtrSetImpl<mlir::Operation*> &visited) {
  mlir::Operation *defOp = val.getDefiningOp();
  if (!defOp || !visited.insert(defOp).second)
    return;

  // 检查是否是 PRB 操作（到达叶子节点）
  if (mlir::isa<circt::llhd::PrbOp>(defOp)) {
    TracedSignal sig = traceToSignal(val);
    if (sig.isValid()) {
      deps.push_back(sig);
    }
    return;
  }

  // 递归追溯所有操作数
  for (mlir::Value operand : defOp->getOperands()) {
    // 先尝试直接追溯
    TracedSignal directSig = traceToSignal(operand);
    if (directSig.isValid()) {
      deps.push_back(directSig);
    } else {
      // 递归追溯
      traceAllDependencies(operand, deps, visited);
    }
  }
}

/// 简化版：获取 Value 依赖的所有信号
inline llvm::SmallVector<TracedSignal, 8> getAllSignalDependencies(mlir::Value val) {
  llvm::SmallVector<TracedSignal, 8> deps;
  llvm::SmallPtrSet<mlir::Operation*, 16> visited;
  traceAllDependencies(val, deps, visited);
  return deps;
}

//===----------------------------------------------------------------------===//
// 信号分类
//===----------------------------------------------------------------------===//

/// 信号类型
enum class SignalType {
  Unknown,
  Clock,          // 时钟信号 (clk, pclk, ...)
  Reset,          // 复位信号 (rst, rst_n, presetn, ...)
  Enable,         // 使能信号 (enable, en, ...)
  Protocol,       // 协议控制信号 (psel, penable, pwrite, ...)
  Data,           // 数据信号
  Status,         // 状态信号
};

/// 根据信号名推断信号类型
inline SignalType classifySignalByName(llvm::StringRef name) {
  // 时钟
  if (name.contains("clk") || name.contains("clock"))
    return SignalType::Clock;

  // 复位
  if (name.contains("rst") || name.contains("reset"))
    return SignalType::Reset;

  // 使能
  if (name == "enable" || name == "en" || name.ends_with("_en"))
    return SignalType::Enable;

  // APB 协议
  if (name == "psel" || name == "penable" || name == "pwrite" ||
      name == "pready" || name.starts_with("paddr") || name.starts_with("pwdata"))
    return SignalType::Protocol;

  // 状态
  if (name.contains("status") || name.contains("state") || name.contains("flag"))
    return SignalType::Status;

  return SignalType::Data;
}

/// 获取信号类型的字符串表示
inline const char* signalTypeToString(SignalType type) {
  switch (type) {
    case SignalType::Clock:    return "Clock";
    case SignalType::Reset:    return "Reset";
    case SignalType::Enable:   return "Enable";
    case SignalType::Protocol: return "Protocol";
    case SignalType::Data:     return "Data";
    case SignalType::Status:   return "Status";
    default:                   return "Unknown";
  }
}

} // namespace signal_tracing

#endif // SIGNAL_TRACING_H

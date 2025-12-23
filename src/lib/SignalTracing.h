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

/// 信号类型（基于名字，已废弃，请使用 SignalRole）
enum class SignalType {
  Unknown,
  Clock,          // 时钟信号 (clk, pclk, ...)
  Reset,          // 复位信号 (rst, rst_n, presetn, ...)
  Enable,         // 使能信号 (enable, en, ...)
  Protocol,       // 协议控制信号 (psel, penable, pwrite, ...)
  Data,           // 数据信号
  Status,         // 状态信号
};

/// 信号拓扑角色（基于使用模式，不依赖名字）
enum class SignalRole {
  Unknown,              // 未知角色
  ModuleInput,          // 模块输入端口 (BlockArgument)
  ControlFlow,          // 控制流信号 (只用于 cf.cond_br 条件)
  AddressSelector,      // 地址选择 (只用于 icmp 比较)
  DataTransfer,         // 数据传输 (用于 drv 的 value)
  InternalIntermediate, // 内部中间值 (有 drv 写入，也有 prb 读取)
  RegisterState,        // 寄存器状态 (有 drv 写入，多次 prb 读取)
  Constant,             // 常量
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

/// 获取信号角色的字符串表示
inline const char* signalRoleToString(SignalRole role) {
  switch (role) {
    case SignalRole::ModuleInput:          return "ModuleInput";
    case SignalRole::ControlFlow:          return "ControlFlow";
    case SignalRole::AddressSelector:      return "AddressSelector";
    case SignalRole::DataTransfer:         return "DataTransfer";
    case SignalRole::InternalIntermediate: return "InternalIntermediate";
    case SignalRole::RegisterState:        return "RegisterState";
    case SignalRole::Constant:             return "Constant";
    default:                               return "Unknown";
  }
}

//===----------------------------------------------------------------------===//
// 信号拓扑角色分析（方案1：基于使用模式）
//===----------------------------------------------------------------------===//

/// 信号使用统计
struct SignalUsageStats {
  mlir::Value signal;
  llvm::StringRef name;

  // 使用次数统计
  unsigned drvCount = 0;        // 被 drv 写入的次数
  unsigned prbCount = 0;        // 被 prb 读取的次数
  unsigned condBrCount = 0;     // 用于 cond_br 条件的次数
  unsigned icmpCount = 0;       // 用于 icmp 比较的次数
  unsigned muxCondCount = 0;    // 用于 mux 条件的次数
  unsigned arithmeticCount = 0; // 用于算术运算的次数

  // 特殊属性
  bool isBlockArgument = false; // 是否是 BlockArgument (模块输入)
  bool isConstant = false;      // 是否是常量
  bool hasDrv = false;          // 是否有 drv 写入
  bool hasPrb = false;          // 是否有 prb 读取

  SignalUsageStats() = default;
  SignalUsageStats(mlir::Value sig, llvm::StringRef n)
    : signal(sig), name(n) {}
};

/// 分析信号的所有使用方式
inline SignalUsageStats analyzeSignalUsage(mlir::Value signal) {
  SignalUsageStats stats;
  stats.signal = signal;

  // 获取信号名
  if (auto defOp = signal.getDefiningOp()) {
    if (auto nameAttr = defOp->getAttrOfType<mlir::StringAttr>("name")) {
      stats.name = nameAttr.getValue();
    }
  }

  // 检查是否是 BlockArgument (模块输入)
  if (mlir::isa<mlir::BlockArgument>(signal)) {
    stats.isBlockArgument = true;
    return stats;  // 模块输入不需要进一步分析
  }

  // 检查是否是常量
  if (auto constOp = signal.getDefiningOp<circt::hw::ConstantOp>()) {
    stats.isConstant = true;
    return stats;
  }

  // 遍历所有使用者
  for (mlir::Operation *user : signal.getUsers()) {
    // 1. 检查是否用于 drv (写入)
    if (mlir::isa<circt::llhd::DrvOp>(user)) {
      stats.drvCount++;
      stats.hasDrv = true;
    }

    // 2. 检查是否用于 prb (读取)
    else if (mlir::isa<circt::llhd::PrbOp>(user)) {
      stats.prbCount++;
      stats.hasPrb = true;
    }

    // 3. 检查是否用于条件分支
    else if (auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(user)) {
      if (condBr.getCondition() == signal) {
        stats.condBrCount++;
      }
    }

    // 4. 检查是否用于比较操作
    else if (mlir::isa<circt::comb::ICmpOp>(user)) {
      stats.icmpCount++;
    }

    // 5. 检查是否用于 mux 条件
    else if (auto muxOp = mlir::dyn_cast<circt::comb::MuxOp>(user)) {
      if (muxOp.getCond() == signal) {
        stats.muxCondCount++;
      }
    }

    // 6. 检查是否用于算术运算
    else if (mlir::isa<circt::comb::AddOp>(user) ||
             mlir::isa<circt::comb::SubOp>(user) ||
             mlir::isa<circt::comb::MulOp>(user)) {
      stats.arithmeticCount++;
    }
  }

  return stats;
}

/// 根据使用统计推断信号角色
inline SignalRole inferSignalRole(const SignalUsageStats &stats) {
  // 1. 模块输入端口
  if (stats.isBlockArgument) {
    return SignalRole::ModuleInput;
  }

  // 2. 常量
  if (stats.isConstant) {
    return SignalRole::Constant;
  }

  // 3. 控制流信号：只用于条件分支，不用于数据传输
  if (stats.condBrCount > 0 && stats.drvCount == 0 && stats.arithmeticCount == 0) {
    return SignalRole::ControlFlow;
  }

  // 4. 地址选择：只用于 icmp 比较
  if (stats.icmpCount > 0 && stats.condBrCount == 0 && stats.drvCount == 0) {
    return SignalRole::AddressSelector;
  }

  // 5. 数据传输：用于 drv 的 value
  if (stats.hasDrv && !stats.hasPrb) {
    return SignalRole::DataTransfer;
  }

  // 6. 寄存器状态：有写入，多次读取
  if (stats.hasDrv && stats.prbCount > 1) {
    return SignalRole::RegisterState;
  }

  // 7. 内部中间值：有写入和读取
  if (stats.hasDrv && stats.hasPrb) {
    return SignalRole::InternalIntermediate;
  }

  return SignalRole::Unknown;
}

/// 分析信号的拓扑角色（一步到位）
inline SignalRole analyzeSignalRole(mlir::Value signal) {
  SignalUsageStats stats = analyzeSignalUsage(signal);
  return inferSignalRole(stats);
}

//===----------------------------------------------------------------------===//
// 控制流追踪
//===----------------------------------------------------------------------===//

/// 驱动动作：记录一个 llhd.drv 操作的信息
struct DriveAction {
  llvm::StringRef signalName;     // 被驱动的信号名
  mlir::Value signal;             // 被驱动的信号
  mlir::Value value;              // 驱动的值
  bool isConstant;                // 值是否为常量
  llvm::APInt constantValue;      // 如果是常量，具体值

  bool isValid() const { return !signalName.empty(); }
};

/// 条件控制信号：记录触发条件中的控制信号
struct ControlSignal {
  llvm::StringRef name;           // 信号名
  bool requiredValue;             // 触发需要的值 (true=1, false=0)
};

/// 条件分支信息
struct BranchCondition {
  llvm::SmallVector<ControlSignal, 4> controls;  // 控制信号列表
  bool hasPwrite;                 // 是否包含 pwrite
  bool pwriteValue;               // pwrite 需要的值
};

/// 从 AND 操作分析控制条件
inline BranchCondition analyzeAndCondition(circt::comb::AndOp andOp) {
  BranchCondition result;
  result.hasPwrite = false;
  result.pwriteValue = true;

  llvm::SmallDenseMap<llvm::StringRef, bool> seen;

  for (mlir::Value operand : andOp.getOperands()) {
    TracedSignal traced = traceToSignal(operand);
    if (!traced.isValid())
      continue;
    if (seen.count(traced.name))
      continue;

    bool requiredVal = !traced.isInverted;

    if (traced.name == "pwrite") {
      result.hasPwrite = true;
      result.pwriteValue = requiredVal;
      seen.insert({traced.name, requiredVal});
      continue;
    }

    ControlSignal ctrl;
    ctrl.name = traced.name;
    ctrl.requiredValue = requiredVal;
    result.controls.push_back(ctrl);
    seen.insert({traced.name, requiredVal});
  }

  return result;
}

/// 从 Block 中收集所有 llhd.drv 操作
inline void collectDriveOps(mlir::Block *block,
                            llvm::SmallVectorImpl<DriveAction> &actions,
                            llvm::SmallPtrSetImpl<mlir::Block*> &visited,
                            unsigned maxDepth = 10) {
  if (!block || !visited.insert(block).second)
    return;
  if (visited.size() > maxDepth)
    return;

  for (mlir::Operation &op : *block) {
    if (auto drv = mlir::dyn_cast<circt::llhd::DrvOp>(&op)) {
      DriveAction action;

      // 获取目标信号名
      mlir::Value target = drv.getSignal();
      if (auto sigOp = target.getDefiningOp()) {
        if (auto nameAttr = sigOp->getAttrOfType<mlir::StringAttr>("name")) {
          action.signalName = nameAttr.getValue();
        }
      }
      action.signal = target;
      action.value = drv.getValue();

      // 检查值是否为常量
      if (auto constOp = drv.getValue().getDefiningOp<circt::hw::ConstantOp>()) {
        action.isConstant = true;
        action.constantValue = constOp.getValue();
      } else {
        action.isConstant = false;
      }

      if (action.isValid())
        actions.push_back(action);
    }

    // 跟踪无条件分支
    if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(op)) {
      collectDriveOps(br.getDest(), actions, visited, maxDepth);
    }

    // 跟踪条件分支（两个方向都收集）
    if (auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(op)) {
      collectDriveOps(condBr.getTrueDest(), actions, visited, maxDepth);
      collectDriveOps(condBr.getFalseDest(), actions, visited, maxDepth);
    }
  }
}

/// 从条件分支追踪 true 分支的所有驱动操作
inline llvm::SmallVector<DriveAction, 8>
traceTrueBranchDrives(mlir::cf::CondBranchOp condBr, unsigned maxDepth = 10) {
  llvm::SmallVector<DriveAction, 8> actions;
  llvm::SmallPtrSet<mlir::Block*, 16> visited;
  collectDriveOps(condBr.getTrueDest(), actions, visited, maxDepth);
  return actions;
}

/// 从条件分支追踪 false 分支的所有驱动操作
inline llvm::SmallVector<DriveAction, 8>
traceFalseBranchDrives(mlir::cf::CondBranchOp condBr, unsigned maxDepth = 10) {
  llvm::SmallVector<DriveAction, 8> actions;
  llvm::SmallPtrSet<mlir::Block*, 16> visited;
  collectDriveOps(condBr.getFalseDest(), actions, visited, maxDepth);
  return actions;
}

/// 控制流路径信息
struct ControlFlowPath {
  BranchCondition condition;                    // 触发条件
  llvm::SmallVector<DriveAction, 8> actions;    // 该路径上的驱动动作
  bool isTrueBranch;                            // 是 true 分支还是 false 分支
};

/// 分析一个 AND 条件控制的完整路径
inline std::optional<ControlFlowPath>
analyzeControlPath(circt::comb::AndOp andOp) {
  // 检查 AND 结果是否用于条件分支
  if (!andOp.getResult().hasOneUse())
    return std::nullopt;

  mlir::Operation *user = *andOp.getResult().getUsers().begin();
  auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(user);
  if (!condBr)
    return std::nullopt;

  ControlFlowPath path;
  path.condition = analyzeAndCondition(andOp);
  path.actions = traceTrueBranchDrives(condBr);
  path.isTrueBranch = true;

  return path;
}

//===----------------------------------------------------------------------===//
// 地址提取
//===----------------------------------------------------------------------===//

/// 从 block 中提取地址比较值（基于操作类型，不依赖信号名）
/// 查找模式: icmp(extract(signal, offset), constant)
inline std::optional<uint32_t> extractAddressFromBlock(mlir::Block *block) {
  if (!block)
    return std::nullopt;

  for (mlir::Operation &op : *block) {
    // 查找比较操作
    if (auto icmp = mlir::dyn_cast<circt::comb::ICmpOp>(&op)) {
      mlir::Value lhs = icmp.getLhs();
      mlir::Value rhs = icmp.getRhs();

      // 尝试两种顺序: extract vs constant
      auto tryExtract = [](mlir::Value extractVal, mlir::Value constVal) -> std::optional<uint32_t> {
        // 检查是否是 extract 操作
        if (auto extract = extractVal.getDefiningOp<circt::comb::ExtractOp>()) {
          // 检查另一侧是否是常量
          if (auto constOp = constVal.getDefiningOp<circt::hw::ConstantOp>()) {
            uint32_t addrVal = constOp.getValue().getZExtValue();
            // 根据 extract 的 lowBit 计算字节地址
            // 通常 APB 地址从 bit 2 开始提取 (paddr[6:2])
            uint32_t lowBit = extract.getLowBit();
            uint32_t byteAddr = addrVal << lowBit;  // 例如: addrVal << 2
            return byteAddr;
          }
        }
        return std::nullopt;
      };

      // 尝试 lhs = extract, rhs = const
      if (auto addr = tryExtract(lhs, rhs))
        return addr;

      // 尝试 rhs = extract, lhs = const
      if (auto addr = tryExtract(rhs, lhs))
        return addr;
    }
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// 寄存器驱动信息
//===----------------------------------------------------------------------===//

/// 寄存器驱动信息
struct RegisterDriveInfo {
  llvm::StringRef targetSignal;     // 被驱动的信号名
  mlir::Value targetValue;          // 被驱动的信号 Value
  mlir::Value sourceValue;          // 驱动的值
  TracedSignal tracedSource;        // 追踪后的源信号
  bool isConstant;                  // 是否是常量驱动
  llvm::APInt constantValue;        // 如果是常量，具体值

  bool isValid() const { return !targetSignal.empty(); }

  /// 检查是否是写使能信号（基于使用模式，方案1）
  bool isWriteEnable() const {
    // 方案1：基于信号角色分析
    SignalRole targetRole = analyzeSignalRole(targetValue);

    // 写使能信号的特征：
    // 1. 是内部中间值（有 drv 写入，也有 prb 读取）
    // 2. 被读取后用作 mux 的条件（控制寄存器是否更新）
    if (targetRole == SignalRole::InternalIntermediate) {
      // 进一步检查：是否被用作 mux 条件
      for (mlir::Operation *user : targetValue.getUsers()) {
        if (auto prbOp = mlir::dyn_cast<circt::llhd::PrbOp>(user)) {
          // 检查 prb 的结果是否用作 mux 条件
          mlir::Value prbResult = prbOp.getResult();
          for (mlir::Operation *prbUser : prbResult.getUsers()) {
            if (auto muxOp = mlir::dyn_cast<circt::comb::MuxOp>(prbUser)) {
              if (muxOp.getCond() == prbResult) {
                return true;  // 确实用作写使能
              }
            }
          }
        }
      }
    }

    // 如果是常量驱动（true/false），很可能是写使能控制
    if (isConstant)
      return true;

    // 向后兼容：检查名字（临时保留）
    if (targetSignal.contains("_wen"))
      return true;

    return false;
  }

  /// 获取真实寄存器名（去除 _wen 后缀）
  std::string getRealRegisterName() const {
    std::string name = targetSignal.str();

    // 去除 _wen 后缀
    if (name.size() > 4 && name.substr(name.size() - 4) == "_wen") {
      return name.substr(0, name.size() - 4);
    }

    return name;
  }
};

/// 从 block 中收集所有寄存器驱动操作
inline llvm::SmallVector<RegisterDriveInfo, 8>
collectRegisterDrives(mlir::Block *block) {
  llvm::SmallVector<RegisterDriveInfo, 8> drives;

  if (!block)
    return drives;

  for (mlir::Operation &op : *block) {
    if (auto drv = mlir::dyn_cast<circt::llhd::DrvOp>(&op)) {
      RegisterDriveInfo info;
      info.targetValue = drv.getSignal();
      info.sourceValue = drv.getValue();

      // 获取目标信号名
      if (auto sigOp = info.targetValue.getDefiningOp()) {
        if (auto nameAttr = sigOp->getAttrOfType<mlir::StringAttr>("name")) {
          info.targetSignal = nameAttr.getValue();
        }
      }

      // 追踪源信号
      info.tracedSource = traceToSignal(info.sourceValue);

      // 检查是否是常量
      if (auto constOp = info.sourceValue.getDefiningOp<circt::hw::ConstantOp>()) {
        info.isConstant = true;
        info.constantValue = constOp.getValue();
      } else {
        info.isConstant = false;
      }

      if (info.isValid())
        drives.push_back(info);
    }
  }

  return drives;
}

//===----------------------------------------------------------------------===//
// APB 协议模式检测
//===----------------------------------------------------------------------===//

/// APB 控制模式信息
struct APBControlPattern {
  circt::comb::AndOp andOp;                   // AND 操作
  llvm::SmallVector<TracedSignal, 4> operands; // 操作数（追踪后）
  mlir::cf::CondBranchOp branchOp;            // 条件分支
  mlir::Block *writePath;                      // 写路径（可能）
  mlir::Block *readPath;                       // 读路径（可能）

  bool isValid() const { return andOp && branchOp; }

  /// 检查操作数中是否包含某个名字的信号（用于向后兼容）
  bool hasSignalNamed(llvm::StringRef name) const {
    for (const auto &sig : operands) {
      if (sig.name == name)
        return true;
    }
    return false;
  }

  /// 检查是否是 APB 协议模式（基于拓扑角色，方案1）
  bool looksLikeAPBControl() const {
    // 至少需要 3 个操作数
    if (operands.size() < 3)
      return false;

    // 方案1：基于信号角色分析
    // APB 控制模式的特征：
    // 1. 所有操作数都是 ModuleInput（来自模块外部）
    // 2. AND 结果用于控制流分支（这个已经在 detectAPBControl 中检查了）
    // 3. 两个分支路径中都有地址检查（icmp）和寄存器驱动（drv）

    unsigned moduleInputCount = 0;
    unsigned controlFlowCount = 0;

    for (const auto &sig : operands) {
      SignalRole role = analyzeSignalRole(sig.signal);

      if (role == SignalRole::ModuleInput) {
        moduleInputCount++;
      } else if (role == SignalRole::ControlFlow) {
        controlFlowCount++;
      }
    }

    // APB 模式：通常有 3 个模块输入信号进行 AND 操作
    // (psel, penable, pwrite 都是来自外部的控制信号)
    if (moduleInputCount >= 3) {
      return true;
    }

    // 向后兼容：也检查名字（临时保留，逐步迁移）
    bool hasPsel = hasSignalNamed("psel");
    bool hasPenable = hasSignalNamed("penable");
    bool hasPwrite = hasSignalNamed("pwrite");

    return hasPsel && hasPenable && hasPwrite;
  }
};

/// 检测 AND 操作是否是 APB 控制模式
inline std::optional<APBControlPattern>
detectAPBControl(circt::comb::AndOp andOp) {
  APBControlPattern pattern;
  pattern.andOp = andOp;

  // 1. 追踪所有操作数
  for (mlir::Value operand : andOp.getOperands()) {
    TracedSignal sig = traceToSignal(operand);
    if (sig.isValid()) {
      pattern.operands.push_back(sig);
    }
  }

  // 2. 检查 AND 结果是否用于条件分支
  if (!andOp.getResult().hasOneUse())
    return std::nullopt;

  mlir::Operation *user = *andOp.getResult().getUsers().begin();
  auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(user);
  if (!condBr)
    return std::nullopt;

  pattern.branchOp = condBr;
  pattern.writePath = condBr.getTrueDest();
  pattern.readPath = condBr.getFalseDest();

  // 3. 检查是否看起来像 APB 控制
  if (!pattern.looksLikeAPBControl())
    return std::nullopt;

  return pattern;
}

/// 从 APB 控制模式中提取寄存器映射
/// 返回: {地址, 寄存器名, 是否可写}
struct APBRegisterMapping {
  uint32_t address;
  std::string registerName;
  bool isWritable;
  bool isReadable;
};

inline llvm::SmallVector<APBRegisterMapping, 8>
extractAPBRegistersFromPattern(const APBControlPattern &pattern) {
  llvm::SmallVector<APBRegisterMapping, 8> mappings;

  // 1. 分析写路径
  if (pattern.writePath) {
    auto writeAddr = extractAddressFromBlock(pattern.writePath);
    if (writeAddr) {
      auto drives = collectRegisterDrives(pattern.writePath);
      for (const auto &drive : drives) {
        if (drive.isWriteEnable()) {
          APBRegisterMapping mapping;
          mapping.address = *writeAddr;
          mapping.registerName = drive.getRealRegisterName();
          mapping.isWritable = true;
          mapping.isReadable = true;  // 默认可读写都支持
          mappings.push_back(mapping);
        }
      }
    }
  }

  // 2. 分析读路径
  // TODO: 添加只读寄存器的提取逻辑

  return mappings;
}

//===----------------------------------------------------------------------===//
// 方案2: 基于使用模式识别（不依赖信号名字）
//===----------------------------------------------------------------------===//

/// 检查 Block 中是否包含地址比较模式
/// 地址比较模式: icmp eq (extract signal), constant
inline bool hasAddressComparePattern(mlir::Block *block) {
  if (!block) return false;

  for (mlir::Operation &op : *block) {
    if (auto icmp = mlir::dyn_cast<circt::comb::ICmpOp>(&op)) {
      // 检查是否是等于比较
      if (icmp.getPredicate() != circt::comb::ICmpPredicate::eq)
        continue;

      // 检查是否有一个操作数是 extract，另一个是常量
      auto checkExtractAndConst = [](mlir::Value a, mlir::Value b) -> bool {
        bool hasExtract = a.getDefiningOp<circt::comb::ExtractOp>() != nullptr;
        bool hasConst = b.getDefiningOp<circt::hw::ConstantOp>() != nullptr;
        return hasExtract && hasConst;
      };

      if (checkExtractAndConst(icmp.getLhs(), icmp.getRhs()) ||
          checkExtractAndConst(icmp.getRhs(), icmp.getLhs())) {
        return true;
      }
    }
  }
  return false;
}

/// 检查 Block 中是否包含寄存器驱动模式
/// 寄存器驱动模式: drv signal, value (值来自另一个特定信号)
inline bool hasRegisterDrivePattern(mlir::Block *block) {
  if (!block) return false;

  for (mlir::Operation &op : *block) {
    if (mlir::isa<circt::llhd::DrvOp>(&op)) {
      return true;
    }
  }
  return false;
}

/// 递归检查分支路径是否包含 APB 读写模式
inline bool checkBranchForAPBPattern(mlir::Block *block,
                                      llvm::SmallPtrSetImpl<mlir::Block*> &visited,
                                      unsigned maxDepth = 5) {
  if (!block || !visited.insert(block).second || maxDepth == 0)
    return false;

  bool hasIcmp = hasAddressComparePattern(block);
  bool hasDrv = hasRegisterDrivePattern(block);

  if (hasIcmp && hasDrv)
    return true;

  // 继续追踪后续分支
  if (auto *terminator = block->getTerminator()) {
    if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(terminator)) {
      if (checkBranchForAPBPattern(br.getDest(), visited, maxDepth - 1))
        return true;
    }
    if (auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
      if (checkBranchForAPBPattern(condBr.getTrueDest(), visited, maxDepth - 1))
        return true;
      if (checkBranchForAPBPattern(condBr.getFalseDest(), visited, maxDepth - 1))
        return true;
    }
  }

  return false;
}

/// 方案2: 基于使用模式识别 APB 控制（不依赖名字）
/// APB 协议的结构特征：
/// 1. 三个信号进行 AND 操作
/// 2. AND 结果用于 cf::CondBranchOp
/// 3. 两个分支中都有：
///    - comb::ICmpOp (地址比较)
///    - llhd::DrvOp (寄存器驱动)
/// 4. 分支中的 icmp 比较一个 extract 操作和一个常量
/// 5. 分支中的 drv 目标是某些信号，源是另一个特定信号
inline bool isAPBControlByUsagePattern(circt::comb::AndOp andOp) {
  // 1. 检查 AND 操作数数量 >= 3
  if (andOp.getNumOperands() < 3)
    return false;

  // 2. 检查 AND 结果是否用于 cf::CondBranchOp
  if (!andOp.getResult().hasOneUse())
    return false;

  mlir::Operation *user = *andOp.getResult().getUsers().begin();
  auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(user);
  if (!condBr)
    return false;

  // 3. 检查两个分支是否都包含 APB 读写模式
  llvm::SmallPtrSet<mlir::Block*, 16> visitedTrue, visitedFalse;

  bool trueHasPattern = checkBranchForAPBPattern(condBr.getTrueDest(),
                                                  visitedTrue);
  bool falseHasPattern = checkBranchForAPBPattern(condBr.getFalseDest(),
                                                   visitedFalse);

  // 至少一个分支需要有 APB 模式（写路径或读路径）
  return trueHasPattern || falseHasPattern;
}

/// 方案2: 基于使用模式识别时钟信号（不依赖名字）
/// 时钟信号的结构特征：
/// 1. 通过 prb 读取，且 prb 结果在 wait 敏感列表中
/// 2. 自身不被 drv 修改（时钟是输入，不是输出）
/// 3. prb 结果主要用于控制流（边沿检测）
/// 注意：wait 的观察值可能是在 process 外部定义的 prb 结果
inline bool isClockSignalByUsagePattern(mlir::Value signal,
                                         circt::llhd::ProcessOp processOp) {
  if (!processOp)
    return false;

  // 获取信号对应的 llhd.sig 操作
  auto sigOp = signal.getDefiningOp<circt::llhd::SignalOp>();
  if (!sigOp)
    return false;

  // 获取信号名用于调试
  std::string sigName = "unknown";
  if (auto nameAttr = sigOp->getAttrOfType<mlir::StringAttr>("name")) {
    sigName = nameAttr.getValue().str();
  }

  // 0. 时钟信号必须是单比特 (i1)
  // 多比特信号（如 paddr, pwdata）不是时钟
  mlir::Type sigType = sigOp.getType();
  if (auto inoutType = mlir::dyn_cast<circt::hw::InOutType>(sigType)) {
    mlir::Type elementType = inoutType.getElementType();
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
      if (intType.getWidth() != 1) {
        // 多比特信号，不是时钟
        return false;
      }
    }
  }

  // 1. 检查 wait 敏感列表中是否有读取此信号的 prb 结果
  // 注意：prb 可能在 process 外部定义，所以需要追踪 observed 值的来源
  bool isInWaitSensitivity = false;
  processOp.walk([&](circt::llhd::WaitOp waitOp) {
    for (mlir::Value observed : waitOp.getObserved()) {
      // 追踪 observed 值的定义操作
      if (auto prbOp = observed.getDefiningOp<circt::llhd::PrbOp>()) {
        // 检查 prb 读取的是否是目标信号
        if (prbOp.getSignal() == signal) {
          isInWaitSensitivity = true;
          return mlir::WalkResult::interrupt();
        }
      }
    }
    return mlir::WalkResult::advance();
  });

  if (!isInWaitSensitivity) {
    // 不输出这个，太多了
    return false;
  }

  // 2. 检查是否有非输入端口连接的 drv 操作
  // 注意：LLHD 中输入端口通过 `llhd.drv %sig, %input_port` 连接到内部信号
  // 这种 drv 是正常的端口连接，不是逻辑驱动，应该允许
  bool hasLogicDrv = false;
  if (auto parentOp = processOp->getParentOfType<circt::hw::HWModuleOp>()) {
    parentOp.walk([&](circt::llhd::DrvOp drvOp) {
      if (drvOp.getSignal() == signal) {
        // 检查驱动值是否来自 BlockArgument（输入端口）
        mlir::Value drvValue = drvOp.getValue();
        if (!mlir::isa<mlir::BlockArgument>(drvValue)) {
          // 不是输入端口连接，是逻辑驱动
          hasLogicDrv = true;
          return mlir::WalkResult::interrupt();
        }
      }
      return mlir::WalkResult::advance();
    });
  }

  if (hasLogicDrv) {
    return false;
  }

  // 到这里，信号满足基本时钟特征：
  // - 单比特
  // - 在敏感列表中
  // - 只有端口连接（无逻辑驱动）
  //
  // 但复位信号也可能满足这些条件！
  // 需要进一步检查触发效果来区分时钟和复位：
  // - 时钟：触发的所有 drv 都是 hold 模式（保持原值）
  // - 复位：触发的 drv 有状态修改（如 counter = 0）

  // 注意：这里我们先返回 true 标记为候选时钟信号
  // 后续的 isClockByTriggerEffect 会进一步验证
  // 这个分步设计是为了在调用者那里可以选择使用哪种检测策略
  return true;
}

//===----------------------------------------------------------------------===//
// 触发分支效果分析（用于区分时钟和复位信号）
//===----------------------------------------------------------------------===//

/// 触发分支效果分析结果
struct TriggerBranchEffect {
  bool hasAnyDrv = false;           // 分支中是否有任何 drv 操作
  bool allDrvsAreHold = true;       // 所有 drv 是否都是 hold 模式
  bool hasStateModification = false; // 是否有状态修改（非 hold）
  unsigned holdCount = 0;           // hold 操作数量
  unsigned modifyCount = 0;         // 修改操作数量

  /// 是否可以过滤（所有操作都是 hold）
  bool canFilter() const {
    return hasAnyDrv && allDrvsAreHold && !hasStateModification;
  }
};

/// 检查 drv 操作是否是 hold 模式（prb signal == signal）
/// hold 模式：reg = prb reg（保持原值）
inline bool isDrvHoldPattern(circt::llhd::DrvOp drvOp) {
  mlir::Value target = drvOp.getSignal();
  mlir::Value value = drvOp.getValue();

  // 检查值是否是对同一信号的 prb
  if (auto prbOp = value.getDefiningOp<circt::llhd::PrbOp>()) {
    if (prbOp.getSignal() == target) {
      return true;  // reg = prb reg，是 hold 模式
    }
  }

  return false;
}

/// 检查 drv 操作是否是常量初始化（如 counter = 0）
inline bool isDrvConstantInit(circt::llhd::DrvOp drvOp) {
  mlir::Value value = drvOp.getValue();
  return value.getDefiningOp<circt::hw::ConstantOp>() != nullptr;
}

/// 递归收集分支中的所有 drv 操作并分析其效果
inline void collectBranchDrvEffects(
    mlir::Block *block,
    TriggerBranchEffect &effect,
    llvm::SmallPtrSetImpl<mlir::Block*> &visited,
    llvm::SmallPtrSetImpl<mlir::Block*> &waitBlocks,
    unsigned maxDepth = 10) {

  if (!block || !visited.insert(block).second)
    return;
  if (visited.size() > maxDepth)
    return;
  if (waitBlocks.count(block))
    return;  // 到达 wait block，停止

  // 遍历 block 中的所有操作
  for (mlir::Operation &op : *block) {
    if (auto drvOp = mlir::dyn_cast<circt::llhd::DrvOp>(&op)) {
      effect.hasAnyDrv = true;

      if (isDrvHoldPattern(drvOp)) {
        effect.holdCount++;
      } else {
        // 不是 hold 模式，是状态修改
        effect.allDrvsAreHold = false;
        effect.hasStateModification = true;
        effect.modifyCount++;
      }
    }
  }

  // 继续跟踪后续分支
  if (auto *terminator = block->getTerminator()) {
    if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(terminator)) {
      collectBranchDrvEffects(br.getDest(), effect, visited, waitBlocks, maxDepth);
    }
    if (auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
      // 两个分支都要分析
      collectBranchDrvEffects(condBr.getTrueDest(), effect, visited, waitBlocks, maxDepth);
      collectBranchDrvEffects(condBr.getFalseDest(), effect, visited, waitBlocks, maxDepth);
    }
  }
}

/// 分析触发信号控制的分支效果
/// 查找由该信号边沿触发的 cf.cond_br，然后分析其分支中的 drv 操作
inline TriggerBranchEffect analyzeTriggerBranchEffects(
    mlir::Value signal,
    circt::llhd::ProcessOp processOp) {

  TriggerBranchEffect effect;

  if (!signal || !processOp)
    return effect;

  // 收集 wait blocks
  llvm::SmallPtrSet<mlir::Block*, 8> waitBlocks;
  processOp.walk([&](circt::llhd::WaitOp waitOp) {
    waitBlocks.insert(waitOp->getBlock());
  });

  // 查找检测该信号边沿的条件分支
  // 边沿检测模式：cf.cond_br (prb signal) 或 cf.cond_br (xor (prb signal), 1)
  processOp.walk([&](mlir::cf::CondBranchOp condBr) {
    mlir::Value cond = condBr.getCondition();
    TracedSignal traced = traceToSignal(cond);

    // 检查条件是否追溯到目标信号
    if (!traced.isValid())
      return;
    if (traced.signal != signal)
      return;

    // 找到了边沿检测分支，分析其 true 和 false 分支
    llvm::SmallPtrSet<mlir::Block*, 16> visitedTrue;
    llvm::SmallPtrSet<mlir::Block*, 16> visitedFalse;

    // 分析 true 分支（通常是上升沿或信号为 1 时）
    collectBranchDrvEffects(condBr.getTrueDest(), effect, visitedTrue, waitBlocks);

    // 分析 false 分支（通常是下降沿或信号为 0 时）
    collectBranchDrvEffects(condBr.getFalseDest(), effect, visitedFalse, waitBlocks);
  });

  return effect;
}

/// 基于触发效果判断信号是否是可过滤的时钟信号
/// 时钟信号特征：触发的所有 drv 操作都是 hold 模式
/// 复位/控制信号特征：触发的 drv 操作有状态修改
inline bool isClockByTriggerEffect(
    mlir::Value signal,
    circt::llhd::ProcessOp processOp) {

  TriggerBranchEffect effect = analyzeTriggerBranchEffects(signal, processOp);

  // 如果没有任何 drv 操作，可能是纯控制信号，不能简单过滤
  if (!effect.hasAnyDrv)
    return false;

  // 如果所有 drv 都是 hold 模式，则是时钟信号，可以过滤
  return effect.canFilter();
}

/// 统计信号的使用模式（跨 process 分析）
struct SignalUsagePatternStats {
  mlir::Value signal;
  unsigned drvCount = 0;           // 被 drv 写入的次数
  unsigned prbCount = 0;           // 被 prb 读取的次数
  unsigned processCount = 0;       // 使用该信号的 process 数量
  bool isModuleInput = false;      // 是否是模块输入端口
  bool hasSingleDrv = false;       // 是否只有一个 drv 写入点
  bool prbFollowsDrv = false;      // prb 是否紧跟在 drv 之后

  llvm::SmallPtrSet<mlir::Operation*, 4> usingProcesses;
};

/// 分析信号的使用模式（用于判断内部信号）
inline SignalUsagePatternStats analyzeSignalUsagePattern(
    mlir::Value signal, circt::hw::HWModuleOp moduleOp) {
  SignalUsagePatternStats stats;
  stats.signal = signal;

  // 检查是否是模块输入端口
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(signal)) {
    stats.isModuleInput = true;
    return stats;  // 输入端口不是内部信号
  }

  // 遍历模块中的所有 process
  moduleOp.walk([&](circt::llhd::ProcessOp processOp) {
    bool usedInThisProcess = false;

    // 检查 drv 操作
    processOp.walk([&](circt::llhd::DrvOp drvOp) {
      if (drvOp.getSignal() == signal) {
        stats.drvCount++;
        usedInThisProcess = true;
      }
    });

    // 检查 prb 操作
    processOp.walk([&](circt::llhd::PrbOp prbOp) {
      if (prbOp.getSignal() == signal) {
        stats.prbCount++;
        usedInThisProcess = true;
      }
    });

    if (usedInThisProcess) {
      stats.processCount++;
      stats.usingProcesses.insert(processOp.getOperation());
    }
  });

  // 判断是否只有一个 drv 写入点
  stats.hasSingleDrv = (stats.drvCount == 1);

  // 检查 prb 是否紧跟在 drv 之后（同一个 block 中）
  if (stats.hasSingleDrv && stats.prbCount > 0) {
    moduleOp.walk([&](circt::llhd::DrvOp drvOp) {
      if (drvOp.getSignal() != signal)
        return;

      mlir::Block *drvBlock = drvOp->getBlock();
      bool foundDrv = false;

      for (mlir::Operation &op : *drvBlock) {
        if (&op == drvOp.getOperation()) {
          foundDrv = true;
          continue;
        }
        if (foundDrv) {
          if (auto prbOp = mlir::dyn_cast<circt::llhd::PrbOp>(&op)) {
            if (prbOp.getSignal() == signal) {
              stats.prbFollowsDrv = true;
              return;
            }
          }
        }
      }
    });
  }

  return stats;
}

/// 方案2: 基于使用模式识别内部信号（不依赖名字）
/// 内部信号的结构特征：
/// 1. 不是模块输入端口
/// 2. 有且仅有一个 drv 写入点
/// 3. 只在特定 process 内部使用（不跨 process）
/// 4. 写入后立即被 prb 读取，用于计算另一个值
inline bool isInternalSignalByUsagePattern(
    mlir::Value signal, circt::hw::HWModuleOp moduleOp) {
  SignalUsagePatternStats stats = analyzeSignalUsagePattern(signal, moduleOp);

  // 1. 不是模块输入端口
  if (stats.isModuleInput)
    return false;

  // 2. 有且仅有一个 drv 写入点
  if (!stats.hasSingleDrv)
    return false;

  // 3. 只在特定 process 内部使用（不跨 process）
  if (stats.processCount > 1)
    return false;

  // 4. 写入后立即被 prb 读取，用于计算另一个值
  // 这是一个强特征，但不是必须的
  // 如果满足 1-3，并且 prb 紧跟 drv，则更确定是内部信号

  // 综合判断：满足 1、2、3 中的任意组合即可认为是内部信号
  // 核心判断：只在一个 process 内使用且只有一个写入点
  return stats.processCount == 1 && stats.hasSingleDrv;
}

/// 综合判断信号类型（基于使用模式，方案2）
enum class SignalTypeByUsage {
  Unknown,
  ClockSignal,      // 时钟信号
  APBControl,       // APB 控制信号
  InternalSignal,   // 内部信号
  RegisterSignal,   // 寄存器信号（状态存储）
  DataSignal,       // 数据信号（输入/输出）
};

/// 基于使用模式推断信号类型
inline SignalTypeByUsage inferSignalTypeByUsage(
    mlir::Value signal,
    circt::hw::HWModuleOp moduleOp,
    circt::llhd::ProcessOp processOp = nullptr) {

  // 1. 检查是否是时钟信号（结构特征 + 触发效果）
  if (processOp && isClockSignalByUsagePattern(signal, processOp) &&
      isClockByTriggerEffect(signal, processOp)) {
    return SignalTypeByUsage::ClockSignal;
  }

  // 2. 检查是否是内部信号
  if (moduleOp && isInternalSignalByUsagePattern(signal, moduleOp)) {
    return SignalTypeByUsage::InternalSignal;
  }

  // 3. 分析基本使用统计
  SignalUsageStats stats = analyzeSignalUsage(signal);

  // 模块输入通常是数据信号或控制信号
  if (stats.isBlockArgument) {
    return SignalTypeByUsage::DataSignal;
  }

  // 有多次 drv 和 prb，可能是寄存器
  if (stats.drvCount > 0 && stats.prbCount > 1) {
    return SignalTypeByUsage::RegisterSignal;
  }

  return SignalTypeByUsage::Unknown;
}

//===----------------------------------------------------------------------===//
// 方案3: 基于数据流依赖分析
//===----------------------------------------------------------------------===//

/// 信号数据流分析结果
struct SignalDataFlowInfo {
  mlir::Value signal;
  llvm::StringRef name;

  // 数据流源头分析
  bool isModuleInputPort = false;     // 来自模块输入端口 (BlockArgument)
  bool sourcedFromAPBControl = false; // 数据来源于 APB 控制路径

  // 数据流去向分析
  bool flowsToRegisterDrv = false;    // 流向寄存器的 drv
  bool usedInAddressCompare = false;  // 用于地址比较 (icmp)
  bool usedInControlFlow = false;     // 用于控制流 (and/or → cond_br)
  bool usedAsMuxCondition = false;    // 用作 mux 的条件选择

  // 使用模式
  unsigned combLogicUseCount = 0;     // 在组合逻辑中的使用次数
  unsigned registerDrvCount = 0;      // 驱动到寄存器的次数
};

/// 追踪数据流源头（向上追踪）
/// 检查一个值的来源是否是：
/// 1. 模块输入端口 (BlockArgument)
/// 2. APB 控制信号 (psel && penable && pwrite)
inline void traceDataFlowSource(mlir::Value val,
                                 SignalDataFlowInfo &info,
                                 llvm::SmallPtrSetImpl<mlir::Operation*> &visited) {
  // 检查是否是 BlockArgument（模块输入端口）
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
    info.isModuleInputPort = true;
    return;
  }

  mlir::Operation *defOp = val.getDefiningOp();
  if (!defOp || !visited.insert(defOp).second)
    return;

  // 检查是否是 prb 操作（读取信号）
  if (auto prbOp = mlir::dyn_cast<circt::llhd::PrbOp>(defOp)) {
    mlir::Value sig = prbOp.getSignal();
    // 继续追踪信号的来源
    traceDataFlowSource(sig, info, visited);
    return;
  }

  // 检查是否是 AND 操作（可能是 APB 控制）
  if (auto andOp = mlir::dyn_cast<circt::comb::AndOp>(defOp)) {
    // 检查 AND 操作数是否包含 APB 控制信号特征
    // APB 控制信号的特征：多个模块输入的 AND
    unsigned moduleInputCount = 0;
    for (mlir::Value operand : andOp.getOperands()) {
      TracedSignal traced = traceToSignal(operand);
      if (traced.isValid()) {
        // 检查追踪到的信号是否来自输入端口
        if (auto sigOp = traced.signal.getDefiningOp<circt::llhd::SignalOp>()) {
          // 检查是否有 drv 来自 BlockArgument
          for (mlir::Operation *user : sigOp.getResult().getUsers()) {
            if (auto drvOp = mlir::dyn_cast<circt::llhd::DrvOp>(user)) {
              if (mlir::isa<mlir::BlockArgument>(drvOp.getValue())) {
                moduleInputCount++;
                break;
              }
            }
          }
        }
      }
    }
    // 如果 AND 的多个操作数都来自模块输入，可能是 APB 控制
    if (moduleInputCount >= 3) {
      info.sourcedFromAPBControl = true;
    }
    return;
  }

  // 递归追踪其他操作的操作数
  for (mlir::Value operand : defOp->getOperands()) {
    traceDataFlowSource(operand, info, visited);
  }
}

/// 追踪数据流去向（向下追踪）
/// 检查一个信号的值被用于：
/// 1. 驱动寄存器 (drv)
/// 2. 地址比较 (icmp)
/// 3. 控制流 (and/or → cond_br)
/// 4. mux 条件
inline void traceDataFlowDest(mlir::Value signal,
                               SignalDataFlowInfo &info,
                               circt::hw::HWModuleOp moduleOp) {
  if (!moduleOp)
    return;

  // 遍历信号的所有 prb 操作
  moduleOp.walk([&](circt::llhd::PrbOp prbOp) {
    if (prbOp.getSignal() != signal)
      return;

    mlir::Value prbResult = prbOp.getResult();

    // 追踪 prb 结果的使用
    llvm::SmallVector<mlir::Value, 16> worklist;
    llvm::SmallPtrSet<mlir::Operation*, 32> visited;
    worklist.push_back(prbResult);

    while (!worklist.empty()) {
      mlir::Value current = worklist.pop_back_val();

      for (mlir::Operation *user : current.getUsers()) {
        if (!visited.insert(user).second)
          continue;

        // 1. 检查是否流向 drv（驱动寄存器）
        if (auto drvOp = mlir::dyn_cast<circt::llhd::DrvOp>(user)) {
          if (drvOp.getValue() == current) {
            info.flowsToRegisterDrv = true;
            info.registerDrvCount++;
          }
          continue;
        }

        // 2. 检查是否用于 icmp（地址比较）
        if (mlir::isa<circt::comb::ICmpOp>(user)) {
          info.usedInAddressCompare = true;
          continue;
        }

        // 3. 检查是否用于 AND/OR（控制流）
        if (mlir::isa<circt::comb::AndOp>(user) ||
            mlir::isa<circt::comb::OrOp>(user)) {
          // 进一步检查 AND/OR 结果是否用于 cond_br
          mlir::Value andOrResult = user->getResult(0);
          for (mlir::Operation *andOrUser : andOrResult.getUsers()) {
            if (mlir::isa<mlir::cf::CondBranchOp>(andOrUser)) {
              info.usedInControlFlow = true;
            }
          }
          // 继续追踪
          worklist.push_back(andOrResult);
          continue;
        }

        // 4. 检查是否用作 mux 条件
        if (auto muxOp = mlir::dyn_cast<circt::comb::MuxOp>(user)) {
          if (muxOp.getCond() == current) {
            info.usedAsMuxCondition = true;
          }
          continue;
        }

        // 5. 组合逻辑操作（继续追踪）
        if (mlir::isa<circt::comb::ExtractOp>(user) ||
            mlir::isa<circt::comb::ConcatOp>(user) ||
            mlir::isa<circt::comb::XorOp>(user) ||
            mlir::isa<circt::comb::AddOp>(user) ||
            mlir::isa<circt::comb::SubOp>(user)) {
          info.combLogicUseCount++;
          // 继续追踪这些操作的结果
          for (mlir::Value result : user->getResults()) {
            worklist.push_back(result);
          }
        }
      }
    }
  });
}

/// 完整的数据流分析
inline SignalDataFlowInfo analyzeSignalDataFlow(
    mlir::Value signal,
    circt::hw::HWModuleOp moduleOp) {
  SignalDataFlowInfo info;
  info.signal = signal;

  // 获取信号名
  if (auto sigOp = signal.getDefiningOp<circt::llhd::SignalOp>()) {
    if (auto nameAttr = sigOp->getAttrOfType<mlir::StringAttr>("name")) {
      info.name = nameAttr.getValue();
    }
  }

  // 追踪数据流源头
  llvm::SmallPtrSet<mlir::Operation*, 32> visited;
  traceDataFlowSource(signal, info, visited);

  // 追踪数据流去向
  traceDataFlowDest(signal, info, moduleOp);

  return info;
}

/// 方案3: 基于数据流识别 GPIO 输入信号
/// GPIO 输入信号的特征：
/// 1. 是模块输入端口
/// 2. 数据流向: port → prb → 组合逻辑 → drv 内部寄存器
/// 3. 不参与地址选择（不用于 icmp）
/// 4. 不参与控制流（不用于 and/or 形成条件）
inline bool isGPIOInputByDataFlow(mlir::Value signal,
                                   circt::hw::HWModuleOp moduleOp) {
  SignalDataFlowInfo info = analyzeSignalDataFlow(signal, moduleOp);

  // 1. 必须是模块输入端口或来自模块输入
  if (!info.isModuleInputPort) {
    // 检查信号是否通过 drv 从 BlockArgument 连接
    bool connectedFromPort = false;
    if (auto sigOp = signal.getDefiningOp<circt::llhd::SignalOp>()) {
      for (mlir::Operation *user : sigOp.getResult().getUsers()) {
        if (auto drvOp = mlir::dyn_cast<circt::llhd::DrvOp>(user)) {
          if (mlir::isa<mlir::BlockArgument>(drvOp.getValue())) {
            connectedFromPort = true;
            break;
          }
        }
      }
    }
    if (!connectedFromPort)
      return false;
  }

  // 2. 数据流向寄存器 drv（通过组合逻辑）
  if (!info.flowsToRegisterDrv && info.combLogicUseCount == 0)
    return false;

  // 3. 不参与地址选择
  if (info.usedInAddressCompare)
    return false;

  // 4. 不参与控制流
  if (info.usedInControlFlow)
    return false;

  return true;
}

/// 方案3: 基于数据流识别写使能信号
/// 写使能信号 (_wen) 的特征：
/// 1. 数据来源: and(psel, penable, pwrite) → 地址检查 → drv *_wen
/// 2. 使用方式: prb *_wen → 用作 mux 的条件选择
/// 3. 不是最终的寄存器状态（只是控制信号）
inline bool isWriteEnableByDataFlow(mlir::Value signal,
                                     circt::hw::HWModuleOp moduleOp) {
  SignalDataFlowInfo info = analyzeSignalDataFlow(signal, moduleOp);

  // 1. 检查数据来源是否是 APB 控制路径
  // 写使能信号的源头应该是 APB 控制 AND 操作
  bool sourcedFromAPB = false;

  // 检查驱动此信号的值的来源
  moduleOp.walk([&](circt::llhd::DrvOp drvOp) {
    if (drvOp.getSignal() != signal)
      return;

    mlir::Value drvValue = drvOp.getValue();

    // 追踪驱动值的来源
    llvm::SmallPtrSet<mlir::Operation*, 16> visited;

    std::function<bool(mlir::Value)> checkAPBSource = [&](mlir::Value val) -> bool {
      if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val))
        return false;

      mlir::Operation *defOp = val.getDefiningOp();
      if (!defOp || !visited.insert(defOp).second)
        return false;

      // 常量 (true/false) - 写使能设置
      if (mlir::isa<circt::hw::ConstantOp>(defOp))
        return true;

      // AND 操作 - 可能是 APB 控制
      if (mlir::isa<circt::comb::AndOp>(defOp)) {
        // 检查是否用于条件分支（APB 控制特征）
        for (mlir::Operation *user : defOp->getResult(0).getUsers()) {
          if (mlir::isa<mlir::cf::CondBranchOp>(user))
            return true;
        }
      }

      // 递归检查
      for (mlir::Value operand : defOp->getOperands()) {
        if (checkAPBSource(operand))
          return true;
      }

      return false;
    };

    if (checkAPBSource(drvValue)) {
      sourcedFromAPB = true;
    }
  });

  if (!sourcedFromAPB)
    return false;

  // 2. 检查使用方式是否是 mux 条件
  if (!info.usedAsMuxCondition)
    return false;

  // 3. 不应该有大量组合逻辑使用（区别于数据寄存器）
  // 写使能信号主要用于 mux 条件选择，不参与复杂计算

  return true;
}

/// 综合数据流分析结果
enum class SignalTypeByDataFlow {
  Unknown,
  GPIOInput,         // GPIO 外部输入信号
  WriteEnable,       // 写使能信号
  APBProtocol,       // APB 协议信号
  ClockReset,        // 时钟/复位信号
  InternalRegister,  // 内部寄存器
  StateRegister,     // 状态寄存器（可通过 APB 访问）
};

/// 基于数据流推断信号类型
inline SignalTypeByDataFlow inferSignalTypeByDataFlow(
    mlir::Value signal,
    circt::hw::HWModuleOp moduleOp,
    circt::llhd::ProcessOp processOp = nullptr) {

  // 1. 检查是否是 GPIO 输入信号
  if (isGPIOInputByDataFlow(signal, moduleOp)) {
    return SignalTypeByDataFlow::GPIOInput;
  }

  // 2. 检查是否是写使能信号
  if (isWriteEnableByDataFlow(signal, moduleOp)) {
    return SignalTypeByDataFlow::WriteEnable;
  }

  // 3. 检查是否是时钟信号（结构特征 + 触发效果）
  // 只有当触发的所有 drv 都是 hold 模式时才是可过滤的时钟
  // 复位信号虽然结构类似，但有状态修改，不应被归为 ClockReset
  if (processOp && isClockSignalByUsagePattern(signal, processOp) &&
      isClockByTriggerEffect(signal, processOp)) {
    return SignalTypeByDataFlow::ClockReset;
  }

  // 4. 检查是否是内部信号（复用已有方案2）
  if (isInternalSignalByUsagePattern(signal, moduleOp)) {
    return SignalTypeByDataFlow::InternalRegister;
  }

  return SignalTypeByDataFlow::Unknown;
}

/// 获取数据流类型的字符串表示
inline const char* signalTypeByDataFlowToString(SignalTypeByDataFlow type) {
  switch (type) {
    case SignalTypeByDataFlow::GPIOInput:        return "GPIOInput";
    case SignalTypeByDataFlow::WriteEnable:      return "WriteEnable";
    case SignalTypeByDataFlow::APBProtocol:      return "APBProtocol";
    case SignalTypeByDataFlow::ClockReset:       return "ClockReset";
    case SignalTypeByDataFlow::InternalRegister: return "InternalRegister";
    case SignalTypeByDataFlow::StateRegister:    return "StateRegister";
    default:                                     return "Unknown";
  }
}

} // namespace signal_tracing

#endif // SIGNAL_TRACING_H

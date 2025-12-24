#ifndef CLK_ANALYSIS_RESULT_H
#define CLK_ANALYSIS_RESULT_H

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include <string>
#include <vector>
#include <map>

namespace clk_analysis {

/// 寄存器写操作的分类（基于状态变化）
enum class DrvClassification {
  STATE_UNCHANGED,   // 状态不变（hold 或不依赖自身），可过滤
  STATE_ACCUMULATE,  // 状态累加/累减，需要 icount/ptimer
  STATE_LOOP_ITER,   // 循环迭代器，组合逻辑
  STATE_COMPLEX,     // 复杂状态变化，需要进一步分析
};

/// 累积操作的方向
enum class AccumulateDirection {
  NONE,
  UP,    // +const
  DOWN,  // -const
};

/// 信号类型（用于事件回调生成）
enum class SignalRole {
  INTERNAL,    // 内部信号，不暴露 MMIO
  INPUT,       // 输入信号（如 feed, enable），写入触发事件
  OUTPUT,      // 输出信号（如 wdt_reset），读取时计算
  STATE,       // 状态信号（如 counter），需要追踪
};

/// 条件表达式类型
enum class ConditionType {
  NONE,           // 无条件
  SIGNAL_TRUE,    // if (signal)
  SIGNAL_FALSE,   // if (!signal)
  COMPARE_GE,     // if (a >= b)
  COMPARE_LT,     // if (a < b)
  COMPARE_EQ,     // if (a == b)
};

/// 比较类型（用于 ASSIGN_COMPARE）
enum class CompareType {
  NONE,
  GE,   // >=
  LT,   // <
  EQ,   // ==
  NE,   // !=
};

/// 动作类型
enum class ActionType {
  ASSIGN_CONST,    // signal = constant
  ASSIGN_SIGNAL,   // signal = other_signal
  ACCUMULATE,      // signal = signal + const
  ASSIGN_COMPARE,  // signal = (a cmp b)
  COMPUTE,         // signal = expression
};

/// 单个动作（drv 操作的转换）
struct EventAction {
  std::string targetSignal;       // 目标信号名
  ActionType type;                // 动作类型
  int64_t constValue;             // 常量值（用于 ASSIGN_CONST）
  std::string sourceSignal;       // 源信号名（用于 ASSIGN_SIGNAL）
  std::string expression;         // 表达式字符串（用于 COMPUTE）

  // 用于 ASSIGN_COMPARE
  CompareType compareType;        // 比较类型
  std::string compareLhs;         // 比较左操作数
  std::string compareRhs;         // 比较右操作数

  EventAction() : type(ActionType::ASSIGN_CONST), constValue(0),
                  compareType(CompareType::NONE) {}
};

/// 条件分支
struct ConditionalBranch {
  ConditionType condType;         // 条件类型
  std::string condSignal;         // 条件信号名
  std::string compareSignal;      // 比较信号名（用于 COMPARE_*）
  std::vector<EventAction> actions;  // 满足条件时的动作
  std::vector<ConditionalBranch> nestedBranches;  // 嵌套的条件分支

  ConditionalBranch() : condType(ConditionType::NONE) {}
};

/// 事件处理器（一个输入信号的写入回调）
struct EventHandler {
  std::string triggerSignal;      // 触发信号名
  std::vector<ConditionalBranch> branches;  // 条件分支列表

  EventHandler() {}
};

/// 单个信号的分析结果
struct SignalAnalysisResult {
  std::string name;              // 信号名
  int bitWidth;                  // 位宽
  DrvClassification classification;  // 分类
  AccumulateDirection direction;  // 累积方向（仅对 ACCUMULATE 有效）
  int stepValue;                 // 步进值（仅对 ACCUMULATE 有效）
  SignalRole role;               // 信号角色
  std::vector<std::string> dependencies;  // 依赖的信号列表
  std::vector<EventAction> preGeneratedActions;  // 预生成的动作（分类时生成）
  bool hasComplexExpression;     // 是否有无法生成的复杂表达式

  SignalAnalysisResult()
      : name(""), bitWidth(32), classification(DrvClassification::STATE_UNCHANGED),
        direction(AccumulateDirection::NONE), stepValue(0),
        role(SignalRole::INTERNAL), hasComplexExpression(false) {}
};

/// 派生信号表达式类型
enum class DerivedExprType {
  NONE,
  SHIFT_RIGHT,   // signal >> const
  SHIFT_LEFT,    // signal << const
  DIVIDE,        // signal / const
  MULTIPLY,      // signal * const
};

/// 派生信号（由其他信号计算得到）
struct DerivedSignal {
  std::string name;           // 信号名
  int bitWidth;               // 位宽
  std::string sourceSignal;   // 源信号名
  DerivedExprType exprType;   // 表达式类型
  int exprValue;              // 表达式参数（如移位量）

  DerivedSignal() : bitWidth(32), exprType(DerivedExprType::NONE), exprValue(0) {}
};

/// 控制信号关系（输入信号控制计数器启停）
struct ControlRelation {
  std::string controlSignal;   // 控制信号名（如 enable）
  std::string counterSignal;   // 被控制的计数器名（如 counter）
  bool activeHigh;             // true: 信号为1时启动; false: 信号为0时启动

  ControlRelation() : activeHigh(true) {}
};

/// APB 寄存器映射（从 paddr 提取）
struct APBRegisterMapping {
  uint32_t address;            // 字节地址（paddr << 2）
  std::string registerName;    // 寄存器名
  int bitWidth;                // 位宽
  bool isWritable;             // 是否可写（有 APB write 条件）
  bool isReadable;             // 是否可读（有 APB read 条件）

  // 写入特征（用于地址冲突处理）
  bool writeUsesExtract;       // 写入值是否使用 extract(pwdata, bit)
  int writeExtractBit;         // 如果 writeUsesExtract，提取的起始位
  bool isW1C;                  // 是否是 Write-1-to-Clear 模式

  APBRegisterMapping()
      : address(0), bitWidth(32), isWritable(false), isReadable(true),
        writeUsesExtract(false), writeExtractBit(0), isW1C(false) {}
};

/// 地址冲突信息
struct AddressConflict {
  uint32_t address;                              // 冲突的地址
  std::vector<std::string> registerNames;        // 该地址上的所有寄存器名
  std::vector<int> bitWidths;                    // 各寄存器的位宽
  std::vector<bool> isReadable;                  // 各寄存器是否可读
  std::vector<bool> isWritable;                  // 各寄存器是否可写
  std::vector<bool> writeUsesExtract;            // 各寄存器写入是否使用 extract
  std::vector<int> writeExtractBits;             // 各寄存器的 extract 起始位

  /// 推荐的处理模式
  enum class ResolveMode {
    BIT_FIELD,        // 方式1: 位域提取（不同位有不同功能）
    READ_WRITE_ASYM,  // 方式2: 读写不对称（读返回一个值，写影响另一个）
    COMBINED,         // 混合模式
    UNKNOWN           // 无法自动判断
  };
  ResolveMode recommendedMode;

  AddressConflict() : address(0), recommendedMode(ResolveMode::UNKNOWN) {}
};

/// 组合逻辑赋值（process 外部的 llhd.drv）
/// 用于生成 update_state() 函数中的组合逻辑表达式
struct CombinationalAssignment {
  std::string targetSignal;    // 目标信号名
  std::string expression;      // C 表达式字符串
  int bitWidth;                // 信号位宽

  CombinationalAssignment() : bitWidth(32) {}
  CombinationalAssignment(const std::string &target, const std::string &expr, int width)
      : targetSignal(target), expression(expr), bitWidth(width) {}
};

/// 模块分析结果
struct ModuleAnalysisResult {
  std::string moduleName;
  std::vector<SignalAnalysisResult> signals;
  std::vector<EventHandler> eventHandlers;  // 事件处理器列表
  std::map<std::string, std::string> inputSignals;  // 输入信号: name -> type
  std::vector<DerivedSignal> derivedSignals;  // 派生信号列表
  std::vector<ControlRelation> controlRelations;  // 控制信号关系
  std::vector<APBRegisterMapping> apbMappings;  // APB 寄存器地址映射
  std::vector<CombinationalAssignment> combinationalLogic;  // 组合逻辑赋值
  std::vector<AddressConflict> addressConflicts;  // 地址冲突信息
};

/// 分析 LLHD 模块，返回信号分类结果
/// 返回第一个 hw.module 的分析结果
ModuleAnalysisResult analyzeModule(mlir::ModuleOp mod);

/// 分析 LLHD 模块，包括事件处理逻辑
/// 返回完整的分析结果，包含事件处理器
ModuleAnalysisResult analyzeModuleWithEvents(mlir::ModuleOp mod);

//===----------------------------------------------------------------------===//
// 表达式生成检查（用于统一分类和代码生成）
//===----------------------------------------------------------------------===//

/// 尝试为一个 drv 操作生成动作
/// 如果表达式无法生成，返回 COMPUTE 类型且 expression 为 "/* complex expression */"
EventAction tryGenerateAction(circt::llhd::DrvOp drv);

/// 检查动作是否是复杂表达式（无法生成代码）
bool isComplexAction(const EventAction &action);

} // namespace clk_analysis

#endif // CLK_ANALYSIS_RESULT_H

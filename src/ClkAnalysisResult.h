#ifndef CLK_ANALYSIS_RESULT_H
#define CLK_ANALYSIS_RESULT_H

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include <string>
#include <vector>
#include <map>

namespace clk_analysis {

/// 寄存器写操作的分类
enum class DrvClassification {
  CLK_IGNORABLE,   // 可忽略时钟，转换为事件驱动
  CLK_ACCUMULATE,  // 累计型，需要 icount/ptimer
  CLK_LOOP_ITER,   // for 循环迭代器，组合逻辑
  CLK_COMPLEX,     // 复杂依赖，需要进一步分析
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
      : name(""), bitWidth(32), classification(DrvClassification::CLK_IGNORABLE),
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

/// 模块分析结果
struct ModuleAnalysisResult {
  std::string moduleName;
  std::vector<SignalAnalysisResult> signals;
  std::vector<EventHandler> eventHandlers;  // 事件处理器列表
  std::map<std::string, std::string> inputSignals;  // 输入信号: name -> type
  std::vector<DerivedSignal> derivedSignals;  // 派生信号列表
  std::vector<ControlRelation> controlRelations;  // 控制信号关系
};

/// 分析 LLHD 模块，返回信号分类结果
/// 返回第一个 hw.module 的分析结果
ModuleAnalysisResult analyzeModule(mlir::ModuleOp mod);

/// 分析 LLHD 模块，包括事件处理逻辑
/// 返回完整的分析结果，包含事件处理器
ModuleAnalysisResult analyzeModuleWithEvents(mlir::ModuleOp mod);

} // namespace clk_analysis

#endif // CLK_ANALYSIS_RESULT_H

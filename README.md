# LLHD 到 QEMU 设备转换工具

本目录包含从 LLHD IR 自动生成的 QEMU 设备代码。

## 更新日志

### 2024-12-16

**新增功能：**
- **CombTranslator**: 完整的 comb dialect 到 C 表达式翻译器
  - 支持算术操作: add, sub, mul, divs, divu, mods, modu
  - 支持位运算: and, or, xor, shl, shrs, shru
  - 支持比较操作: icmp (eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge)
  - 支持数据操作: extract, concat, replicate, mux, reverse, parity

**修复问题：**
1. **信号名称净化**: 自动将 `.`、`[`、`]`、`-`、空格替换为 `_`，确保生成有效的 C 标识符
   - 例如: `SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2` → `SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff2`

2. **冗余条件消除**: 消除嵌套条件中的重复检查
   - 之前: `if (value) { if (value) { ... } }`
   - 之后: `if (value) { ... }`

3. **未定义信号处理**: 对于无法找到的信号，生成注释而非无效代码
   - 输出: `/* undefined signal: signal_name */`

4. **COMPUTE 动作修复**: 正确生成计算表达式
   - 添加 `s->` 前缀
   - 添加分号
   - 例如: `s->unnamed = (((s->int_level) >> (arg0)) & 1);`

5. **统一事件处理**: `collectDrvsInBlock()` 现在使用 `tryGenerateAction()` 而非旧的 `analyzeDrvAction()`，确保所有表达式都经过 CombTranslator 处理

## 已知问题

### 1. BlockArgument 未解析 (arg0, arg1, ...)

LLHD 中的循环迭代变量会被翻译为 `arg0`、`arg1` 等占位符：

```c
// 生成的代码
s->unnamed = (((s->int_level) >> (arg0)) & 1);
```

**原因**: 这来自 Verilog 的 generate/for 循环结构，例如：
```verilog
for (i = 0; i < 32; i = i + 1)
    result[i] = gpio_int_level_sync ? int_level[i] : int_level_sync_in[i];
```

**解决方案**: 需要手动替换 `arg0` 为具体值，或展开为位操作

### 2. 部分信号未定义

某些内部临时信号可能未被添加到状态结构体：
- `gpio_int_clk_en_tmp`
- `int_level`

**解决方案**: 检查原始 Verilog 并手动添加缺失的信号定义

### 3. Generate/For 循环展开

目前不支持自动展开硬件循环为软件循环或位操作序列

## 使用方法

```bash
# 分析 LLHD IR 并显示信号分类结果
./dff-opt input.mlir --analyze-clk

# 生成 QEMU 设备代码
./dff-opt input.mlir --gen-qemu --qemu-out=/path/to/output
```

## 信号分类逻辑

工具分析每个 `llhd.drv` 操作，将信号分为 4 类：

### 1. CLK_IGNORABLE（事件驱动型）

不依赖自身的信号，可转换为简单的寄存器写入。

```verilog
// 示例：简单赋值
always @(posedge clk)
    reg_a <= data_in;
```

**QEMU 输出**：事件处理器中的直接赋值
```c
s->reg_a = s->data_in;
```

### 2. CLK_ACCUMULATE（计数器型）

具有 `signal = signal +/- constant` 形式的自依赖信号。

```verilog
// 示例：计数器
always @(posedge clk)
    counter <= counter + 1;
```

**QEMU 输出**：基于 ptimer 的计数器，使用 QEMU_CLOCK_VIRTUAL
```c
ptimer_state *counter_ptimer;
// 使用 ptimer_get_count() / ptimer_set_count()
```

### 3. CLK_LOOP_ITER（循环迭代器）

组合逻辑中的 for 循环迭代变量。

```verilog
// 示例：for 循环迭代器
for (i = 0; i < 32; i = i + 1)
    result[i] <= data[i] & mask[i];
```

**QEMU 输出**：展开为位操作（目前跳过）

### 4. CLK_COMPLEX（需手动处理）

表达式无法转换为 C 代码的信号。

**判定为 CLK_COMPLEX 的条件**：
- 不是常量赋值
- 不是简单信号赋值
- 不是比较结果赋值
- 不是自累加模式
- CombTranslator 无法翻译的操作

**QEMU 输出**：跳过并警告，需要手动实现

## 表达式生成（与分类统一）

工具采用统一方案，分类与代码生成耦合：

```
对每个 drv 操作：
  1. 尝试生成 C 表达式 (tryGenerateAction)
  2. 如果生成失败 -> CLK_COMPLEX
  3. 如果生成成功 -> 根据模式分类
```

### 支持的表达式类型

| 类型 | 模式 | C 代码 |
|------|------|--------|
| ASSIGN_CONST | `sig <= 0` | `s->sig = 0;` |
| ASSIGN_SIGNAL | `sig <= other` | `s->sig = s->other;` |
| ASSIGN_COMPARE | `sig <= (a >= b)` | `s->sig = (s->a >= s->b);` |
| ACCUMULATE | `sig <= sig + 1` | `ptimer_set_count(...)` |
| COMPUTE | comb 操作 | `s->sig = expr;` |

### CombTranslator 支持的操作

| 操作类别 | 操作 | C 表达式 |
|----------|------|----------|
| 算术 | add, sub, mul | `+`, `-`, `*` |
| 算术 | divu, divs | `/` (有符号用 cast) |
| 算术 | modu, mods | `%` |
| 位运算 | and, or, xor | `&`, `\|`, `^` |
| 位运算 | shl, shru, shrs | `<<`, `>>` |
| 数据操作 | extract | `(val >> lowBit) & mask` |
| 数据操作 | concat | `(hi << width) \| lo` |
| 数据操作 | mux | `cond ? true : false` |
| 数据操作 | replicate | `bit ? 0xFFFF : 0` |
| 比较 | icmp | `==`, `!=`, `<`, `<=`, `>`, `>=` |

## 事件处理器生成

为每个触发状态变化的输入信号生成：

```c
static void device_on_signal_write(device_state *s, uint32_t value)
{
    if (value) {        // SIGNAL_TRUE 条件
        s->reg = ...;
    }
    if (!value) {       // SIGNAL_FALSE 条件
        s->reg = 0;
    }
}
```

## 控制信号检测

工具检测输入信号与计数器之间的控制关系：

```verilog
// 如果 enable=1 触发计数器累加
always @(posedge clk)
    if (enable)
        counter <= counter + 1;
```

**检测结果**：`enable` 控制 `counter`（activeHigh=true）

## 生成的文件

为每个模块生成两个文件：

- `module_name.h` - 设备状态结构体和类型声明
- `module_name.c` - 设备实现，包含：
  - 内存映射寄存器读写
  - 输入信号的事件处理器
  - 计数器的 ptimer 设置
  - 复位逻辑

## 目录结构

```
qemu-output/
├── README.md           # 本文档
├── gpio0/              # GPIO 设备生成示例
│   ├── gpio_top.h      # 设备头文件
│   └── gpio_top.c      # 设备实现
├── src/                # 源代码
│   └── lib/            # 库文件
│       ├── CombTranslator.h    # Comb 操作翻译器
│       ├── ClkAnalysisResult.h # 信号分析结果定义
│       ├── QEMUCodeGen.h       # QEMU 代码生成器
│       └── SignalTracing.h     # 信号追踪工具
└── test/               # 测试文件
```

## 架构说明

```
┌─────────────────┐
│   Verilog/SV    │
└────────┬────────┘
         │ moore (转换)
         ▼
┌─────────────────┐
│    LLHD IR      │
└────────┬────────┘
         │ dff-opt --gen-qemu
         ▼
┌─────────────────────────────────────────┐
│            信号分类阶段                   │
│  ┌─────────────────────────────────┐    │
│  │ tryGenerateAction()             │    │
│  │ - 尝试生成 C 表达式              │    │
│  │ - 使用 CombTranslator 翻译      │    │
│  │ - 成功 → 继续分类                │    │
│  │ - 失败 → CLK_COMPLEX            │    │
│  └─────────────────────────────────┘    │
│                 ▼                        │
│  ┌─────────────────────────────────┐    │
│  │ 模式匹配分类                     │    │
│  │ - 不依赖自己 → CLK_IGNORABLE    │    │
│  │ - sig+const → CLK_ACCUMULATE    │    │
│  │ - for 循环 → CLK_LOOP_ITER      │    │
│  └─────────────────────────────────┘    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│            代码生成阶段                   │
│  - CLK_IGNORABLE → 简单寄存器           │
│  - CLK_ACCUMULATE → ptimer 计数器       │
│  - COMPUTE → CombTranslator 表达式      │
│  - CLK_COMPLEX → 跳过（需手动）          │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────┐
│  QEMU C 代码    │
│  (.h + .c)      │
└─────────────────┘
```

## 示例输出

### gpio0 模块

生成的 `gpio_top.h`:
```c
typedef struct gpio_top_state {
    SysBusDevice parent_obj;
    MemoryRegion iomem;
    qemu_irq irq;

    uint32_t gpio_int_en;
    uint8_t gpio_int_level_sync;
    // ... 44 个信号

    /* ptimer-based counter: int_k */
    ptimer_state *int_k_ptimer;
    uint32_t int_k_limit;

    uint32_t presetn;  /* input */
    uint32_t pwdata;   /* input */
    // ... 27 个输入信号
} gpio_top_state;
```

生成的事件处理器:
```c
static void gpio_top_on_presetn_write(gpio_top_state *s, uint32_t value)
{
    if (!value) {
        s->gpio_int_en = 0;
    }
    if (value) {
        s->gpio_int_en = s->pwdata;
    }
    // ... 更多条件分支
}
```

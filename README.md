# LLHD 到 QEMU 设备转换工具

本目录包含从 LLHD IR 自动生成的 QEMU 设备代码。

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
| COMPUTE | （复杂表达式） | `/* complex expression */` |

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

## 当前限制

1. **复杂表达式**：位操作、多路选择器、优先编码器需要手动处理
2. **for 循环展开**：位级操作的循环展开尚未实现
3. **信号命名**：包含 `.` 的名称（如 `PROC.signal`）会生成无效的 C 标识符
4. **嵌套条件**：深层嵌套可能产生冗余的条件检查

## 示例

输入（Verilog）：
```verilog
module counter(input clk, input rst_n, input enable, output [31:0] count);
    reg [31:0] counter;
    always @(posedge clk or negedge rst_n)
        if (!rst_n)
            counter <= 0;
        else if (enable)
            counter <= counter + 1;
    assign count = counter;
endmodule
```

输出（QEMU）：
```c
// counter.h
typedef struct counter_state {
    SysBusDevice parent_obj;
    MemoryRegion iomem;
    qemu_irq irq;
    ptimer_state *counter_ptimer;
    uint32_t counter_limit;
    uint32_t enable;  /* input */
} counter_state;

// counter.c
static void counter_on_enable_write(counter_state *s, uint32_t value)
{
    if (value) {
        counter_start_counter(s);  // 启动 ptimer
    }
    if (!value) {
        counter_stop_counter(s);   // 停止 ptimer
    }
}
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
│  - CLK_COMPLEX → 跳过（需手动）          │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────┐
│  QEMU C 代码    │
│  (.h + .c)      │
└─────────────────┘
```

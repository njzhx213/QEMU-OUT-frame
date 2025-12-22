========================================
QEMU Code Generation (from LLHD analysis)
========================================

Analyzed module: gpio_top
Found 63 signals
Found 27 input signals
Found 2 event handlers

Signal Classifications:
  - ri_gpio_raw_int_status (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_int_en_wen (i1): CLK_IGNORABLE -> SimpleReg
  - gpio_int_clr (i32): CLK_IGNORABLE -> SimpleReg
  - ri_gpio_sw_data (i32): CLK_IGNORABLE -> SimpleReg
  - zero_value (i1): CLK_IGNORABLE -> SimpleReg
  - gpio_int_clk_en (i1): CLK_IGNORABLE -> SimpleReg
  - gpio_int_type (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_int_clr (i32): CLK_IGNORABLE -> SimpleReg
  - SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2 (i32): CLK_IGNORABLE -> SimpleReg
  - prdata (i32): CLK_COMPLEX -> skip (needs manual handling)
  - gpio_int_clr_wen (i1): CLK_IGNORABLE -> SimpleReg
  - ri_gpio_int_en (i32): CLK_IGNORABLE -> SimpleReg
  - int_edge_out (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_sw_dir (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_int_clk_en_tmp (i1): CLK_IGNORABLE -> SimpleReg
  - pclk_int (i1): CLK_IGNORABLE -> SimpleReg
  - ri_gpio_sw_dir (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_int_flag_tmp (i1): CLK_IGNORABLE -> SimpleReg
  - SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff1 (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_int_status_level (i32): CLK_IGNORABLE -> SimpleReg
  - int_level (i32): CLK_IGNORABLE -> SimpleReg
  - ri_gpio_int_mask (i32): CLK_IGNORABLE -> SimpleReg
  - int_level_ff1 (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_int_status (i32): CLK_IGNORABLE -> SimpleReg
  - penable (i1): CLK_IGNORABLE -> SimpleReg
  - gpio_rx_data_int (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_int_en (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_sw_data_wen (i1): CLK_IGNORABLE -> SimpleReg
  - ri_gpio_int_status (i32): CLK_IGNORABLE -> SimpleReg
  - int_level_sync_in (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_int_type_wen (i1): CLK_IGNORABLE -> SimpleReg
  - gpio_sw_dir_wen (i1): CLK_IGNORABLE -> SimpleReg
  - gpio_rx_data (i32): CLK_IGNORABLE -> SimpleReg
  - presetn (i1): CLK_IGNORABLE -> SimpleReg
  - gpio_ext_data (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_raw_int_status (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_int_level_sync (i1): CLK_IGNORABLE -> SimpleReg
  - ri_gpio_int_type (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_int_pol (i32): CLK_IGNORABLE -> SimpleReg
  - pwrite (i1): CLK_IGNORABLE -> SimpleReg
  - gpio_int_level_sync_wen (i1): CLK_IGNORABLE -> SimpleReg
  - gpio_tx_data (i1): CLK_IGNORABLE -> SimpleReg
  - pclk (i1): CLK_IGNORABLE -> SimpleReg
  - gpio_debounce_wen (i1): CLK_IGNORABLE -> SimpleReg
  - ri_gpio_debounce (i32): CLK_IGNORABLE -> SimpleReg
  - ri_gpio_int_level_sync (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_int_status_edge (i32): CLK_IGNORABLE -> SimpleReg
  - int_clk_en (i1): CLK_IGNORABLE -> SimpleReg
  - gpio_sw_data (i32): CLK_IGNORABLE -> SimpleReg
  - pwdata (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_int_mask_wen (i1): CLK_IGNORABLE -> SimpleReg
  - psel (i1): CLK_IGNORABLE -> SimpleReg
  - int_k (i32): CLK_IGNORABLE -> SimpleReg
  - paddr (i7): CLK_IGNORABLE -> SimpleReg
  - gpio_int_pol_wen (i1): CLK_IGNORABLE -> SimpleReg
  - int_edge (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_tx_en (i1): CLK_IGNORABLE -> SimpleReg
  - ri_gpio_int_pol (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_int_mask (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_ext_data_tmp (i32): CLK_IGNORABLE -> SimpleReg
  - debounce_d2 (i32): CLK_IGNORABLE -> SimpleReg
  - gpio_debounce (i32): CLK_IGNORABLE -> SimpleReg
  - ri_gpio_ext_data (i32): CLK_IGNORABLE -> SimpleReg

Input Signals (classified):
  - dbclk [clock] -> filtered (not needed in simulation)
  - dbclk_rstn [clock] -> filtered (not needed in simulation)
  - gpio_debounce [input] -> event trigger
  - gpio_ext_data [gpio_in] -> qdev_init_gpio_in
  - gpio_ext_porta [gpio_in] -> qdev_init_gpio_in
  - gpio_in_data [gpio_in] -> qdev_init_gpio_in
  - gpio_int_clr [input] -> event trigger
  - gpio_int_en [input] -> event trigger
  - gpio_int_level_sync [input] -> event trigger
  - gpio_int_mask [input] -> event trigger
  - gpio_int_pol [input] -> event trigger
  - gpio_int_status [input] -> event trigger
  - gpio_int_type [input] -> event trigger
  - gpio_raw_int_status [input] -> event trigger
  - gpio_rx_data [input] -> event trigger
  - gpio_sw_data [input] -> event trigger
  - gpio_sw_dir [input] -> event trigger
  - paddr [apb] -> MMIO read/write
  - pclk [clock] -> filtered (not needed in simulation)
  - pclk_int [clock] -> filtered (not needed in simulation)
  - pclk_intr [clock] -> filtered (not needed in simulation)
  - penable [apb] -> MMIO read/write
  - presetn [input] -> event trigger
  - psel [apb] -> MMIO read/write
  - pwdata [apb] -> MMIO read/write
  - pwrite [apb] -> MMIO read/write
  - scan_mode [input] -> event trigger

Event Handlers:
  - on_presetn_write: 25 branches
  - on_gpio_int_level_sync_write: 2 branches

  - 0x00: gpio_sw_data (R/W)
  - 0x04: gpio_sw_dir (R/W)
  - 0x30: gpio_int_en (R/W)
  - 0x34: gpio_int_mask (R/W)
  - 0x38: gpio_int_type (R/W)
  - 0x3c: gpio_int_pol (R/W)
  - 0x40: gpio_int_status (R)
  - 0x44: gpio_raw_int_status (R)
  - 0x48: gpio_debounce (R)
  - 0x50: gpio_ext_data (R)
  - 0x60: gpio_int_level_sync (R/W)

/* ==================== gpio_top.h ==================== */

/*
 * Auto-generated QEMU device: gpio_top
 * Generated by LLHD-to-QEMU converter
 *
 * Uses ptimer + QEMU_CLOCK_VIRTUAL for counter implementation.
 * Compatible with both icount mode and real-time mode.
 */

#ifndef HW_GPIO_TOP_H
#define HW_GPIO_TOP_H

#include "hw/sysbus.h"
#include "hw/ptimer.h"
#include "qom/object.h"
#include "qemu/timer.h"

#define TYPE_GPIO_TOP "gpio_top"
OBJECT_DECLARE_SIMPLE_TYPE(gpio_top_state, GPIO_TOP)

typedef struct gpio_top_state {
    SysBusDevice parent_obj;
    MemoryRegion iomem;
    qemu_irq irq;

    uint32_t ri_gpio_raw_int_status;
    uint8_t gpio_int_en_wen;
    uint32_t gpio_int_clr;
    uint32_t ri_gpio_sw_data;
    uint8_t zero_value;
    uint8_t gpio_int_clk_en;
    uint32_t gpio_int_type;
    uint32_t gpio_int_clr;
    uint32_t SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff2;
    uint8_t gpio_int_clr_wen;
    uint32_t ri_gpio_int_en;
    uint32_t int_edge_out;
    uint32_t gpio_sw_dir;
    uint8_t gpio_int_clk_en_tmp;
    uint8_t pclk_int;
    uint32_t ri_gpio_sw_dir;
    uint8_t gpio_int_flag_tmp;
    uint32_t SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff1;
    uint32_t gpio_int_status_level;
    uint32_t int_level;
    uint32_t ri_gpio_int_mask;
    uint32_t int_level_ff1;
    uint32_t gpio_int_status;
    uint8_t penable;
    uint32_t gpio_rx_data_int;
    uint32_t gpio_int_en;
    uint8_t gpio_sw_data_wen;
    uint32_t ri_gpio_int_status;
    uint32_t int_level_sync_in;
    uint8_t gpio_int_type_wen;
    uint8_t gpio_sw_dir_wen;
    uint32_t gpio_rx_data;
    uint8_t presetn;
    uint32_t gpio_ext_data;
    uint32_t gpio_raw_int_status;
    uint8_t gpio_int_level_sync;
    uint32_t ri_gpio_int_type;
    uint32_t gpio_int_pol;
    uint8_t pwrite;
    uint8_t gpio_int_level_sync_wen;
    uint8_t gpio_tx_data;
    uint8_t pclk;
    uint8_t gpio_debounce_wen;
    uint32_t ri_gpio_debounce;
    uint32_t ri_gpio_int_level_sync;
    uint32_t gpio_int_status_edge;
    uint8_t int_clk_en;
    uint32_t gpio_sw_data;
    uint32_t pwdata;
    uint8_t gpio_int_mask_wen;
    uint8_t psel;
    uint32_t int_k;
    uint8_t paddr;
    uint8_t gpio_int_pol_wen;
    uint32_t int_edge;
    uint8_t gpio_tx_en;
    uint32_t ri_gpio_int_pol;
    uint32_t gpio_int_mask;
    uint32_t gpio_ext_data_tmp;
    uint32_t debounce_d2;
    uint32_t gpio_debounce;
    uint32_t ri_gpio_ext_data;
    uint32_t scan_mode;  /* input */
    uint32_t gpio_ext_porta;  /* gpio input */
    uint32_t gpio_in_data;  /* gpio input */
} gpio_top_state;

#endif /* HW_GPIO_TOP_H */


/* ==================== gpio_top.c ==================== */

/*
 * Auto-generated QEMU device: gpio_top
 *
 * Uses ptimer with QEMU_CLOCK_VIRTUAL.
 * - In icount mode: virtual clock driven by instruction counter
 * - In real-time mode: virtual clock driven by host timer
 */

#include "qemu/osdep.h"
#include "hw/irq.h"
#include "hw/qdev-properties.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "gpio_top.h"

/*
 * ptimer 到期回调 - 计数器归零时触发
 */
static void gpio_top_timer_tick(void *opaque)
{
    gpio_top_state *s = opaque;
    /* 触发中断 */
    qemu_irq_raise(s->irq);
}

/*
 * Update State - recalculate combinational logic after input changes
 * TODO: Generate from Signal Tracing results
 */
static void gpio_top_update_state(gpio_top_state *s)
{
    /* Combinational logic: gpio_ext_porta -> int_level -> gpio_int_status */
    /* TODO: Add traced combinational expressions here */

    /* Update interrupt output */
    uint32_t pending = s->gpio_int_status & s->gpio_int_en & ~s->gpio_int_mask;
    if (pending) {
        qemu_irq_raise(s->irq);
    } else {
        qemu_irq_lower(s->irq);
    }
}

/*
 * GPIO Input Callback - called when external GPIO state changes
 */
static void gpio_top_gpio_input_set(void *opaque, int line, int value)
{
    gpio_top_state *s = GPIO_TOP(opaque);

    /* Update gpio_ext_data */
    s->gpio_ext_data = deposit32(s->gpio_ext_data, line, 1, value != 0);
    /* Update gpio_ext_porta */
    s->gpio_ext_porta = deposit32(s->gpio_ext_porta, line, 1, value != 0);
    /* Update gpio_in_data */
    s->gpio_in_data = deposit32(s->gpio_in_data, line, 1, value != 0);

    /* Recalculate combinational logic */
    gpio_top_update_state(s);
}

/*
 * Event Handlers - triggered by input signal writes
 */

/*
 * Handler for presetn signal changes
 */
static void gpio_top_on_presetn_write(gpio_top_state *s, uint32_t value)
{
    if (!value) {
        s->gpio_int_en = 0;
    }
    if (value) {
        s->gpio_int_en = s->pwdata;
    }
    if (!value) {
        s->gpio_int_mask = 0;
    }
    if (value) {
        s->gpio_int_mask = s->pwdata;
    }
    if (!value) {
        s->gpio_int_type = 0;
    }
    if (value) {
        s->gpio_int_type = s->pwdata;
    }
    if (!value) {
        s->gpio_int_pol = 0;
    }
    if (value) {
        s->gpio_int_pol = s->pwdata;
    }
    if (!value) {
        s->gpio_int_level_sync = 0;
    }
    if (value) {
        s->gpio_int_level_sync = ((s->pwdata) & 1);
    }
    if (!value) {
        s->gpio_sw_data = 0;
    }
    if (value) {
        s->gpio_sw_data = s->pwdata;
    }
    if (!value) {
        s->gpio_sw_dir = 0;
    }
    if (value) {
        s->gpio_sw_dir = s->pwdata;
    }
    if (!value) {
        s->prdata = 0;
    }
    if (!value) {
        s->gpio_int_clk_en = 0;
    }
    if (value) {
        s->gpio_int_clk_en = s->gpio_int_clk_en_tmp;
    }
    if (!value) {
        s->int_level_ff1 = 0;
    }
    if (value) {
        s->int_level_ff1 = s->int_level;
    }
    if (!value) {
        s->gpio_int_status_edge = 0;
    }
    if (value) {
        s->int_k = 0;
    }
    if (!value) {
        s->SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff1 = 0;
    }
    if (value) {
        s->SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff1 = s->int_level_sync_in;
    }
    if (!value) {
        s->SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff2 = 0;
    }
    if (value) {
        s->SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff2 = s->SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff1;
    }
}

/*
 * Handler for gpio_int_level_sync signal changes
 */
static void gpio_top_on_gpio_int_level_sync_write(gpio_top_state *s, uint32_t value)
{
    if (value) {
        s->gpio_int_status_level = s->int_level;
    }
    if (!value) {
        s->gpio_int_status_level = s->int_level_sync_in;
    }
}

static uint64_t gpio_top_read(void *opaque, hwaddr addr, unsigned size)
{
    gpio_top_state *s = opaque;
    uint64_t value = 0;

    switch (addr) {
    case 0x00:  /* gpio_sw_data */
        value = s->gpio_sw_data;
        break;
    case 0x04:  /* gpio_sw_dir */
        value = s->gpio_sw_dir;
        break;
    case 0x30:  /* gpio_int_en */
        value = s->gpio_int_en;
        break;
    case 0x34:  /* gpio_int_mask */
        value = s->gpio_int_mask;
        break;
    case 0x38:  /* gpio_int_type */
        value = s->gpio_int_type;
        break;
    case 0x3c:  /* gpio_int_pol */
        value = s->gpio_int_pol;
        break;
    case 0x40:  /* gpio_int_status */
        value = s->gpio_int_status;
        break;
    case 0x44:  /* gpio_raw_int_status */
        value = s->gpio_raw_int_status;
        break;
    case 0x48:  /* gpio_debounce */
        value = s->gpio_debounce;
        break;
    case 0x50:  /* gpio_ext_data */
        value = s->gpio_ext_data;
        break;
    case 0x60:  /* gpio_int_level_sync */
        value = s->gpio_int_level_sync;
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR, "gpio_top: bad read at 0x%" HWADDR_PRIx "\n", addr);
    }
    return value;
}

static void gpio_top_write(void *opaque, hwaddr addr,
                              uint64_t value, unsigned size)
{
    gpio_top_state *s = opaque;

    switch (addr) {
    case 0x00:  /* gpio_sw_data */
        s->gpio_sw_data = value;
        break;
    case 0x04:  /* gpio_sw_dir */
        s->gpio_sw_dir = value;
        break;
    case 0x30:  /* gpio_int_en */
        s->gpio_int_en = value;
        break;
    case 0x34:  /* gpio_int_mask */
        s->gpio_int_mask = value;
        break;
    case 0x38:  /* gpio_int_type */
        s->gpio_int_type = value;
        break;
    case 0x3c:  /* gpio_int_pol */
        s->gpio_int_pol = value;
        break;
    case 0x60:  /* gpio_int_level_sync */
        s->gpio_int_level_sync = value;
        gpio_top_on_gpio_int_level_sync_write(s, value);
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR, "gpio_top: bad write at 0x%" HWADDR_PRIx "\n", addr);
    }
}

static const MemoryRegionOps gpio_top_ops = {
    .read = gpio_top_read,
    .write = gpio_top_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
};

static void gpio_top_init(Object *obj)
{
    gpio_top_state *s = GPIO_TOP(obj);
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);

    memory_region_init_io(&s->iomem, obj, &gpio_top_ops, s,
                          TYPE_GPIO_TOP, 0x1000);
    sysbus_init_mmio(sbd, &s->iomem);
    sysbus_init_irq(sbd, &s->irq);

    /* Initialize GPIO inputs */
    qdev_init_gpio_in(DEVICE(s), gpio_top_gpio_input_set, 32);
}

static void gpio_top_realize(DeviceState *dev, Error **errp)
{
    gpio_top_state *s = GPIO_TOP(dev);

}

static void gpio_top_reset(DeviceState *dev)
{
    gpio_top_state *s = GPIO_TOP(dev);

    s->ri_gpio_raw_int_status = 0;
    s->gpio_int_en_wen = 0;
    s->gpio_int_clr = 0;
    s->ri_gpio_sw_data = 0;
    s->zero_value = 0;
    s->gpio_int_clk_en = 0;
    s->gpio_int_type = 0;
    s->gpio_int_clr = 0;
    s->SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff2 = 0;
    s->gpio_int_clr_wen = 0;
    s->ri_gpio_int_en = 0;
    s->int_edge_out = 0;
    s->gpio_sw_dir = 0;
    s->gpio_int_clk_en_tmp = 0;
    s->pclk_int = 0;
    s->ri_gpio_sw_dir = 0;
    s->gpio_int_flag_tmp = 0;
    s->SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff1 = 0;
    s->gpio_int_status_level = 0;
    s->int_level = 0;
    s->ri_gpio_int_mask = 0;
    s->int_level_ff1 = 0;
    s->gpio_int_status = 0;
    s->penable = 0;
    s->gpio_rx_data_int = 0;
    s->gpio_int_en = 0;
    s->gpio_sw_data_wen = 0;
    s->ri_gpio_int_status = 0;
    s->int_level_sync_in = 0;
    s->gpio_int_type_wen = 0;
    s->gpio_sw_dir_wen = 0;
    s->gpio_rx_data = 0;
    s->presetn = 0;
    s->gpio_ext_data = 0;
    s->gpio_raw_int_status = 0;
    s->gpio_int_level_sync = 0;
    s->ri_gpio_int_type = 0;
    s->gpio_int_pol = 0;
    s->pwrite = 0;
    s->gpio_int_level_sync_wen = 0;
    s->gpio_tx_data = 0;
    s->pclk = 0;
    s->gpio_debounce_wen = 0;
    s->ri_gpio_debounce = 0;
    s->ri_gpio_int_level_sync = 0;
    s->gpio_int_status_edge = 0;
    s->int_clk_en = 0;
    s->gpio_sw_data = 0;
    s->pwdata = 0;
    s->gpio_int_mask_wen = 0;
    s->psel = 0;
    s->int_k = 0;
    s->paddr = 0;
    s->gpio_int_pol_wen = 0;
    s->int_edge = 0;
    s->gpio_tx_en = 0;
    s->ri_gpio_int_pol = 0;
    s->gpio_int_mask = 0;
    s->gpio_ext_data_tmp = 0;
    s->debounce_d2 = 0;
    s->gpio_debounce = 0;
    s->ri_gpio_ext_data = 0;
    qemu_irq_lower(s->irq);
}

static void gpio_top_class_init(ObjectClass *oc, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);
    dc->realize = gpio_top_realize;
    dc->reset = gpio_top_reset;
}

static const TypeInfo gpio_top_info = {
    .name = TYPE_GPIO_TOP,
    .parent = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(gpio_top_state),
    .instance_init = gpio_top_init,
    .class_init = gpio_top_class_init,
};

static void gpio_top_register_types(void)
{
    type_register_static(&gpio_top_info);
}

type_init(gpio_top_register_types)

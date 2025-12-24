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
 * Auto-generated from LLHD combinational drv operations
 */
static void gpio_top_update_state(gpio_top_state *s)
{
    /* Combinational logic assignments */
    s->zero_value = 0;
    s->gpio_int_clk_en_tmp = ((s->int_clk_en) != (0));
    s->int_edge = (s->int_level) ^ (s->int_level_ff1);
    s->gpio_int_flag_tmp = ((s->gpio_int_status) != (0));
    s->int_level = ((s->gpio_int_level_sync) ? (s->SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff2) : (s->int_level_sync_in));
    s->gpio_ext_data_tmp = s->gpio_rx_data;

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

    /* Update gpio_rx_data */
    s->gpio_rx_data = deposit32(s->gpio_rx_data, line, 1, value != 0);

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
    case 0x60:  /* CONFLICT: gpio_int_level_sync + gpio_int_clr */
        value = s->gpio_int_level_sync;
        break;
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
    case 0x60:  /* CONFLICT: gpio_int_level_sync + gpio_int_clr */
        s->gpio_int_level_sync = value & 1;  /* 1-bit */
        s->gpio_int_status &= ~value;  /* W1C */
        break;
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

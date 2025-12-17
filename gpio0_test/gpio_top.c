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
    case 0x00:  /* ri_gpio_raw_int_status */
        value = s->ri_gpio_raw_int_status;
        break;
    case 0x04:  /* gpio_int_en_wen */
        value = s->gpio_int_en_wen;
        break;
    case 0x08:  /* gpio_int_clr */
        value = s->gpio_int_clr;
        break;
    case 0x0c:  /* ri_gpio_sw_data */
        value = s->ri_gpio_sw_data;
        break;
    case 0x10:  /* zero_value */
        value = s->zero_value;
        break;
    case 0x14:  /* gpio_int_clk_en */
        value = s->gpio_int_clk_en;
        break;
    case 0x18:  /* gpio_int_type */
        value = s->gpio_int_type;
        break;
    case 0x1c:  /* gpio_int_clr */
        value = s->gpio_int_clr;
        break;
    case 0x20:  /* SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2 */
        value = s->SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff2;
        break;
    case 0x24:  /* gpio_int_clr_wen */
        value = s->gpio_int_clr_wen;
        break;
    case 0x28:  /* ri_gpio_int_en */
        value = s->ri_gpio_int_en;
        break;
    case 0x2c:  /* int_edge_out */
        value = s->int_edge_out;
        break;
    case 0x30:  /* gpio_sw_dir */
        value = s->gpio_sw_dir;
        break;
    case 0x34:  /* gpio_int_clk_en_tmp */
        value = s->gpio_int_clk_en_tmp;
        break;
    case 0x38:  /* pclk_int */
        value = s->pclk_int;
        break;
    case 0x3c:  /* ri_gpio_sw_dir */
        value = s->ri_gpio_sw_dir;
        break;
    case 0x40:  /* gpio_int_flag_tmp */
        value = s->gpio_int_flag_tmp;
        break;
    case 0x44:  /* SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff1 */
        value = s->SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff1;
        break;
    case 0x48:  /* gpio_int_status_level */
        value = s->gpio_int_status_level;
        break;
    case 0x4c:  /* int_level */
        value = s->int_level;
        break;
    case 0x50:  /* ri_gpio_int_mask */
        value = s->ri_gpio_int_mask;
        break;
    case 0x54:  /* int_level_ff1 */
        value = s->int_level_ff1;
        break;
    case 0x58:  /* gpio_int_status */
        value = s->gpio_int_status;
        break;
    case 0x5c:  /* penable */
        value = s->penable;
        break;
    case 0x60:  /* gpio_rx_data_int */
        value = s->gpio_rx_data_int;
        break;
    case 0x64:  /* gpio_int_en */
        value = s->gpio_int_en;
        break;
    case 0x68:  /* gpio_sw_data_wen */
        value = s->gpio_sw_data_wen;
        break;
    case 0x6c:  /* ri_gpio_int_status */
        value = s->ri_gpio_int_status;
        break;
    case 0x70:  /* int_level_sync_in */
        value = s->int_level_sync_in;
        break;
    case 0x74:  /* gpio_int_type_wen */
        value = s->gpio_int_type_wen;
        break;
    case 0x78:  /* gpio_sw_dir_wen */
        value = s->gpio_sw_dir_wen;
        break;
    case 0x7c:  /* gpio_rx_data */
        value = s->gpio_rx_data;
        break;
    case 0x80:  /* presetn */
        value = s->presetn;
        break;
    case 0x84:  /* gpio_ext_data */
        value = s->gpio_ext_data;
        break;
    case 0x88:  /* gpio_raw_int_status */
        value = s->gpio_raw_int_status;
        break;
    case 0x8c:  /* gpio_int_level_sync */
        value = s->gpio_int_level_sync;
        break;
    case 0x90:  /* ri_gpio_int_type */
        value = s->ri_gpio_int_type;
        break;
    case 0x94:  /* gpio_int_pol */
        value = s->gpio_int_pol;
        break;
    case 0x98:  /* pwrite */
        value = s->pwrite;
        break;
    case 0x9c:  /* gpio_int_level_sync_wen */
        value = s->gpio_int_level_sync_wen;
        break;
    case 0xa0:  /* gpio_tx_data */
        value = s->gpio_tx_data;
        break;
    case 0xa4:  /* pclk */
        value = s->pclk;
        break;
    case 0xa8:  /* gpio_debounce_wen */
        value = s->gpio_debounce_wen;
        break;
    case 0xac:  /* ri_gpio_debounce */
        value = s->ri_gpio_debounce;
        break;
    case 0xb0:  /* ri_gpio_int_level_sync */
        value = s->ri_gpio_int_level_sync;
        break;
    case 0xb4:  /* gpio_int_status_edge */
        value = s->gpio_int_status_edge;
        break;
    case 0xb8:  /* int_clk_en */
        value = s->int_clk_en;
        break;
    case 0xbc:  /* gpio_sw_data */
        value = s->gpio_sw_data;
        break;
    case 0xc0:  /* pwdata */
        value = s->pwdata;
        break;
    case 0xc4:  /* gpio_int_mask_wen */
        value = s->gpio_int_mask_wen;
        break;
    case 0xc8:  /* psel */
        value = s->psel;
        break;
    case 0xcc:  /* int_k */
        value = s->int_k;
        break;
    case 0xd0:  /* paddr */
        value = s->paddr;
        break;
    case 0xd4:  /* gpio_int_pol_wen */
        value = s->gpio_int_pol_wen;
        break;
    case 0xd8:  /* int_edge */
        value = s->int_edge;
        break;
    case 0xdc:  /* gpio_tx_en */
        value = s->gpio_tx_en;
        break;
    case 0xe0:  /* ri_gpio_int_pol */
        value = s->ri_gpio_int_pol;
        break;
    case 0xe4:  /* gpio_int_mask */
        value = s->gpio_int_mask;
        break;
    case 0xe8:  /* gpio_ext_data_tmp */
        value = s->gpio_ext_data_tmp;
        break;
    case 0xec:  /* debounce_d2 */
        value = s->debounce_d2;
        break;
    case 0xf0:  /* gpio_debounce */
        value = s->gpio_debounce;
        break;
    case 0xf4:  /* ri_gpio_ext_data */
        value = s->ri_gpio_ext_data;
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
    case 0x00:  /* ri_gpio_raw_int_status */
        s->ri_gpio_raw_int_status = value;
        break;
    case 0x04:  /* gpio_int_en_wen */
        s->gpio_int_en_wen = value;
        break;
    case 0x08:  /* gpio_int_clr */
        s->gpio_int_clr = value;
        break;
    case 0x0c:  /* ri_gpio_sw_data */
        s->ri_gpio_sw_data = value;
        break;
    case 0x10:  /* zero_value */
        s->zero_value = value;
        break;
    case 0x14:  /* gpio_int_clk_en */
        s->gpio_int_clk_en = value;
        break;
    case 0x18:  /* gpio_int_type */
        s->gpio_int_type = value;
        break;
    case 0x1c:  /* gpio_int_clr */
        s->gpio_int_clr = value;
        break;
    case 0x20:  /* SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2 */
        s->SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff2 = value;
        break;
    case 0x24:  /* gpio_int_clr_wen */
        s->gpio_int_clr_wen = value;
        break;
    case 0x28:  /* ri_gpio_int_en */
        s->ri_gpio_int_en = value;
        break;
    case 0x2c:  /* int_edge_out */
        s->int_edge_out = value;
        break;
    case 0x30:  /* gpio_sw_dir */
        s->gpio_sw_dir = value;
        break;
    case 0x34:  /* gpio_int_clk_en_tmp */
        s->gpio_int_clk_en_tmp = value;
        break;
    case 0x38:  /* pclk_int */
        s->pclk_int = value;
        break;
    case 0x3c:  /* ri_gpio_sw_dir */
        s->ri_gpio_sw_dir = value;
        break;
    case 0x40:  /* gpio_int_flag_tmp */
        s->gpio_int_flag_tmp = value;
        break;
    case 0x44:  /* SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff1 */
        s->SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff1 = value;
        break;
    case 0x48:  /* gpio_int_status_level */
        s->gpio_int_status_level = value;
        break;
    case 0x4c:  /* int_level */
        s->int_level = value;
        break;
    case 0x50:  /* ri_gpio_int_mask */
        s->ri_gpio_int_mask = value;
        break;
    case 0x54:  /* int_level_ff1 */
        s->int_level_ff1 = value;
        break;
    case 0x58:  /* gpio_int_status */
        s->gpio_int_status = value;
        break;
    case 0x5c:  /* penable */
        s->penable = value;
        break;
    case 0x60:  /* gpio_rx_data_int */
        s->gpio_rx_data_int = value;
        break;
    case 0x64:  /* gpio_int_en */
        s->gpio_int_en = value;
        break;
    case 0x68:  /* gpio_sw_data_wen */
        s->gpio_sw_data_wen = value;
        break;
    case 0x6c:  /* ri_gpio_int_status */
        s->ri_gpio_int_status = value;
        break;
    case 0x70:  /* int_level_sync_in */
        s->int_level_sync_in = value;
        break;
    case 0x74:  /* gpio_int_type_wen */
        s->gpio_int_type_wen = value;
        break;
    case 0x78:  /* gpio_sw_dir_wen */
        s->gpio_sw_dir_wen = value;
        break;
    case 0x7c:  /* gpio_rx_data */
        s->gpio_rx_data = value;
        break;
    case 0x80:  /* presetn */
        s->presetn = value;
        gpio_top_on_presetn_write(s, value);
        break;
    case 0x84:  /* gpio_ext_data */
        s->gpio_ext_data = value;
        break;
    case 0x88:  /* gpio_raw_int_status */
        s->gpio_raw_int_status = value;
        break;
    case 0x8c:  /* gpio_int_level_sync */
        s->gpio_int_level_sync = value;
        gpio_top_on_gpio_int_level_sync_write(s, value);
        break;
    case 0x90:  /* ri_gpio_int_type */
        s->ri_gpio_int_type = value;
        break;
    case 0x94:  /* gpio_int_pol */
        s->gpio_int_pol = value;
        break;
    case 0x98:  /* pwrite */
        s->pwrite = value;
        break;
    case 0x9c:  /* gpio_int_level_sync_wen */
        s->gpio_int_level_sync_wen = value;
        break;
    case 0xa0:  /* gpio_tx_data */
        s->gpio_tx_data = value;
        break;
    case 0xa4:  /* pclk */
        s->pclk = value;
        break;
    case 0xa8:  /* gpio_debounce_wen */
        s->gpio_debounce_wen = value;
        break;
    case 0xac:  /* ri_gpio_debounce */
        s->ri_gpio_debounce = value;
        break;
    case 0xb0:  /* ri_gpio_int_level_sync */
        s->ri_gpio_int_level_sync = value;
        break;
    case 0xb4:  /* gpio_int_status_edge */
        s->gpio_int_status_edge = value;
        break;
    case 0xb8:  /* int_clk_en */
        s->int_clk_en = value;
        break;
    case 0xbc:  /* gpio_sw_data */
        s->gpio_sw_data = value;
        break;
    case 0xc0:  /* pwdata */
        s->pwdata = value;
        break;
    case 0xc4:  /* gpio_int_mask_wen */
        s->gpio_int_mask_wen = value;
        break;
    case 0xc8:  /* psel */
        s->psel = value;
        break;
    case 0xcc:  /* int_k */
        s->int_k = value;
        break;
    case 0xd0:  /* paddr */
        s->paddr = value;
        break;
    case 0xd4:  /* gpio_int_pol_wen */
        s->gpio_int_pol_wen = value;
        break;
    case 0xd8:  /* int_edge */
        s->int_edge = value;
        break;
    case 0xdc:  /* gpio_tx_en */
        s->gpio_tx_en = value;
        break;
    case 0xe0:  /* ri_gpio_int_pol */
        s->ri_gpio_int_pol = value;
        break;
    case 0xe4:  /* gpio_int_mask */
        s->gpio_int_mask = value;
        break;
    case 0xe8:  /* gpio_ext_data_tmp */
        s->gpio_ext_data_tmp = value;
        break;
    case 0xec:  /* debounce_d2 */
        s->debounce_d2 = value;
        break;
    case 0xf0:  /* gpio_debounce */
        s->gpio_debounce = value;
        break;
    case 0xf4:  /* ri_gpio_ext_data */
        s->ri_gpio_ext_data = value;
        break;
    case 0xf8:  /* dbclk (input) */
        s->dbclk = value;
        break;
    case 0xfc:  /* dbclk_rstn (input) */
        s->dbclk_rstn = value;
        break;
    case 0x100:  /* gpio_ext_porta (input) */
        s->gpio_ext_porta = value;
        break;
    case 0x104:  /* gpio_in_data (input) */
        s->gpio_in_data = value;
        break;
    case 0x108:  /* pclk_intr (input) */
        s->pclk_intr = value;
        break;
    case 0x10c:  /* scan_mode (input) */
        s->scan_mode = value;
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

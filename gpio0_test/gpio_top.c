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

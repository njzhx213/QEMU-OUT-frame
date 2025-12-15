// 看门狗定时器 - 需要定期喂狗，否则产生复位信号
module watchdog_timer #(
    parameter WIDTH = 32
) (
    input  logic clk,
    input  logic rst_n,

    // 控制接口
    input  logic enable,
    input  logic feed,                    // 喂狗信号
    input  logic [WIDTH-1:0] timeout_val, // 超时值

    // 输出
    output logic [WIDTH-1:0] current_count,
    output logic wdt_reset,               // 看门狗复位输出
    output logic wdt_warning              // 接近超时警告
);

    logic [WIDTH-1:0] counter;
    logic [WIDTH-1:0] warning_threshold;

    assign current_count = counter;
    assign warning_threshold = timeout_val >> 1;  // 50% 时发出警告

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter <= '0;
            wdt_reset <= 1'b0;
            wdt_warning <= 1'b0;
        end else if (enable) begin
            if (feed) begin
                // 喂狗：重置计数器
                counter <= '0;
                wdt_reset <= 1'b0;
                wdt_warning <= 1'b0;
            end else if (counter >= timeout_val) begin
                // 超时：产生复位
                wdt_reset <= 1'b1;
            end else begin
                counter <= counter + 1'b1;
                wdt_warning <= (counter >= warning_threshold);
            end
        end else begin
            counter <= '0;
            wdt_reset <= 1'b0;
            wdt_warning <= 1'b0;
        end
    end

endmodule

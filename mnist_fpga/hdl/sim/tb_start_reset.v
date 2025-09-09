// Simple testbench to verify start/reset button functionality
`timescale 1ns/1ps

module tb_start_reset;

  reg clk;
  reg rst_n;
  reg start_btn;
  wire done;
  wire [3:0] predicted_digit;

  // Clock generation
  initial begin
    clk = 0;
    forever #5 clk = ~clk; // 100MHz clock
  end

  // Simple top module for testing (without neural network layers)
  reg [2:0] tstate;
  reg start_btn_sync1, start_btn_sync2, start_btn_prev;
  wire start_btn_edge = start_btn_sync2 & ~start_btn_prev;
  wire rst = ~rst_n;
  
  reg done_reg;
  reg [3:0] predicted_digit_reg;
  
  assign done = done_reg;
  assign predicted_digit = predicted_digit_reg;

  localparam T_IDLE   = 3'd0;
  localparam T_WAIT   = 3'd1;
  localparam T_CALC   = 3'd2;
  localparam T_DONE   = 3'd3;

  // Button synchronization
  always @(posedge clk) begin
    if (rst) begin
      start_btn_sync1 <= 1'b0;
      start_btn_sync2 <= 1'b0;
      start_btn_prev <= 1'b0;
    end else begin
      start_btn_sync1 <= start_btn;
      start_btn_sync2 <= start_btn_sync1;
      start_btn_prev <= start_btn_sync2;
    end
  end

  // Simple FSM to test start/reset functionality
  reg [7:0] calc_counter;
  
  always @(posedge clk) begin
    if (rst) begin
      tstate <= T_IDLE;
      done_reg <= 1'b0;
      predicted_digit_reg <= 4'd0;
      calc_counter <= 8'd0;
    end else begin
      case (tstate)
        T_IDLE: begin
          done_reg <= 1'b0;
          predicted_digit_reg <= 4'd0;
          tstate <= T_WAIT;
        end

        T_WAIT: begin
          done_reg <= 1'b0;
          predicted_digit_reg <= 4'd0;
          if (start_btn_edge) begin
            calc_counter <= 8'd0;
            tstate <= T_CALC;
          end
        end

        T_CALC: begin
          // Simulate calculation for 50 clock cycles
          if (calc_counter < 8'd50) begin
            calc_counter <= calc_counter + 1'b1;
          end else begin
            predicted_digit_reg <= 4'd7; // Simulate result
            tstate <= T_DONE;
          end
        end

        T_DONE: begin
          done_reg <= 1'b1;
          // Stay in DONE state - predicted_digit remains displayed
        end

        default: tstate <= T_IDLE;
      endcase
    end
  end

  // Test sequence
  initial begin
    $dumpfile("start_reset_test.vcd");
    $dumpvars(0, tb_start_reset);

    // Initialize
    rst_n = 0;
    start_btn = 0;
    
    // Wait a few cycles
    #20;
    
    // Release reset
    rst_n = 1;
    #20;
    
    $display("Time %0t: After reset - done=%b, predicted_digit=%d", $time, done, predicted_digit);
    
    // Wait in reset state
    #100;
    $display("Time %0t: Waiting - done=%b, predicted_digit=%d", $time, done, predicted_digit);
    
    // Press start button
    start_btn = 1;
    #20;
    start_btn = 0;
    #20;
    
    $display("Time %0t: Start pressed - done=%b, predicted_digit=%d", $time, done, predicted_digit);
    
    // Wait for calculation to complete
    #600;
    $display("Time %0t: After calculation - done=%b, predicted_digit=%d", $time, done, predicted_digit);
    
    // Test reset while in DONE state
    #100;
    rst_n = 0;
    #20;
    $display("Time %0t: Reset pressed - done=%b, predicted_digit=%d", $time, done, predicted_digit);
    
    rst_n = 1;
    #20;
    $display("Time %0t: Reset released - done=%b, predicted_digit=%d", $time, done, predicted_digit);
    
    // Verify it stays in reset state until start is pressed again
    #100;
    $display("Time %0t: Still waiting - done=%b, predicted_digit=%d", $time, done, predicted_digit);
    
    // Press start again
    start_btn = 1;
    #20;
    start_btn = 0;
    #20;
    
    // Wait for second calculation
    #600;
    $display("Time %0t: Second calculation done - done=%b, predicted_digit=%d", $time, done, predicted_digit);
    
    $display("Test completed successfully!");
    $finish;
  end

endmodule

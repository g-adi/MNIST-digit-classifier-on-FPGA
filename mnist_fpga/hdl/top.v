// hdl/top.v
`timescale 1ns/1ps

module top #(
  // Layer sizes
  parameter integer IN1   = 784,
  parameter integer H1    = 32,
  parameter integer IN2   = 32,
  parameter integer OUT2  = 10
)(
  input  wire clk,
  input  wire rst_n,
  input  wire start_btn,  // Start button input

  // Optional simple outputs for sim/board bring-up
  output reg        done,
  output reg [3:0]  predicted_digit  // fits 0..9
);

  // Reset button synchronization and edge detection
  reg rst_btn_sync1, rst_btn_sync2, rst_btn_prev;
  wire rst_btn_edge = rst_btn_sync2 & ~rst_btn_prev;
  reg system_reset;

  // -------------------------
  // Memories (behavioral arrays for sim & easy bring-up)
  // -------------------------
  // Input image (int8)
  reg signed [7:0] x_mem [0:IN1-1];

  // Layer1 params
  reg signed [7:0]  W1_mem [0:H1*IN1-1];
  reg signed [31:0] b1_mem [0:H1-1];

  // Layer1 output buffer (int8)
  reg signed [7:0]  h1_mem [0:H1-1];

  // Layer2 params
  reg signed [7:0]  W2_mem [0:OUT2*IN2-1];
  reg signed [31:0] b2_mem [0:OUT2-1];

  // Layer2 output (logits int32)
  reg signed [31:0] y2_mem [0:OUT2-1];

  // shift1 parameter (hardcoded from shift1.txt for synthesis)
  localparam [5:0] shift1 = 6'd0;

  // ------------- Memory Initialization for Synthesis -------------
  initial begin
    // Direct file paths for synthesis (no string concatenation)
    $readmemh("W1.mem", W1_mem);
    $readmemh("b1.mem", b1_mem);
    $readmemh("W2.mem", W2_mem);
    $readmemh("b2.mem", b2_mem);
    $readmemh("sample_input.mem", x_mem);
  end

  // -------------------------
  // FC1 instance (writes h1_mem)
  // -------------------------
  // READ hookups
  wire [$clog2(IN1)-1:0]       fc1_x_addr;
  wire signed [7:0]            fc1_x_data = x_mem[fc1_x_addr];

  wire [$clog2(IN1*H1)-1:0]    fc1_w_addr;
  wire signed [7:0]            fc1_w_data = W1_mem[fc1_w_addr];

  wire [$clog2(H1)-1:0]        fc1_b_addr;
  wire signed [31:0]           fc1_b_data = b1_mem[fc1_b_addr];

  // WRITE hookups
  wire                         fc1_y_we;
  wire [$clog2(H1)-1:0]        fc1_y_addr;
  wire signed [7:0]            fc1_y_data;

  always @(posedge clk) begin
    if (fc1_y_we)
      h1_mem[fc1_y_addr] <= fc1_y_data;
  end

  reg fc1_start;
  wire fc1_done;

  fc1_layer #(.IN_DIM(IN1), .OUT_DIM(H1)) u_fc1 (
    .clk(clk), .rst(system_reset), .start(fc1_start), .done(fc1_done),
    .shift_right(shift1),
    .x_addr(fc1_x_addr), .x_data(fc1_x_data),
    .w_addr(fc1_w_addr), .w_data(fc1_w_data),
    .b_addr(fc1_b_addr), .b_data(fc1_b_data),
    .y_we(fc1_y_we), .y_addr(fc1_y_addr), .y_data(fc1_y_data)
  );

  // -------------------------
  // FC2 instance (reads h1_mem, writes y2_mem)
  // -------------------------
  wire [$clog2(IN2)-1:0]       fc2_x_addr;
  wire signed [7:0]            fc2_x_data = h1_mem[fc2_x_addr];

  wire [$clog2(IN2*OUT2)-1:0]  fc2_w_addr;
  wire signed [7:0]            fc2_w_data = W2_mem[fc2_w_addr];

  wire [$clog2(OUT2)-1:0]      fc2_b_addr;
  wire signed [31:0]           fc2_b_data = b2_mem[fc2_b_addr];

  wire                         fc2_y_we;
  wire [$clog2(OUT2)-1:0]      fc2_y_addr;
  wire signed [31:0]           fc2_y_data;

  always @(posedge clk) begin
    if (fc2_y_we)
      y2_mem[fc2_y_addr] <= fc2_y_data;
  end

  reg fc2_start;
  wire fc2_done;

  fc2_layer #(.IN_DIM(IN2), .OUT_DIM(OUT2)) u_fc2 (
    .clk(clk), .rst(system_reset), .start(fc2_start), .done(fc2_done),
    .x_addr(fc2_x_addr), .x_data(fc2_x_data),
    .w_addr(fc2_w_addr), .w_data(fc2_w_data),
    .b_addr(fc2_b_addr), .b_data(fc2_b_data),
    .y_we(fc2_y_we), .y_addr(fc2_y_addr), .y_data(fc2_y_data)
  );

  // -------------------------
  // Top-level FSM + Argmax
  // -------------------------
  localparam T_IDLE   = 3'd0;
  localparam T_WAIT   = 3'd1;
  localparam T_L1     = 3'd2;
  localparam T_L2     = 3'd3;
  localparam T_ARGMAX = 3'd4;
  localparam T_DONE   = 3'd5;

  reg [2:0] tstate;

  // Argmax machinery
  reg [3:0]  idx;
  reg signed [31:0] maxv;
  reg [3:0]  maxi;

  // Button synchronization and edge detection
  reg start_btn_sync1, start_btn_sync2, start_btn_prev;
  wire start_btn_edge = start_btn_sync2 & ~start_btn_prev;

  // Reset button synchronization (always runs, no reset dependency)
  always @(posedge clk) begin
    rst_btn_sync1 <= ~rst_n;  // Invert because button is active low
    rst_btn_sync2 <= rst_btn_sync1;
    rst_btn_prev <= rst_btn_sync2;
  end

  // System reset generation
  always @(posedge clk) begin
    if (rst_btn_edge) begin
      system_reset <= 1'b1;
    end else begin
      system_reset <= 1'b0;
    end
  end

  // Start button synchronization
  always @(posedge clk) begin
    if (system_reset) begin
      start_btn_sync1 <= 1'b0;
      start_btn_sync2 <= 1'b0;
      start_btn_prev <= 1'b0;
    end else begin
      start_btn_sync1 <= start_btn;
      start_btn_sync2 <= start_btn_sync1;
      start_btn_prev <= start_btn_sync2;
    end
  end

  always @(posedge clk) begin
    if (system_reset) begin
      tstate <= T_IDLE;
      fc1_start <= 1'b0;
      fc2_start <= 1'b0;
      done <= 1'b0;
      predicted_digit <= 4'd0;
      idx  <= 4'd0; maxv <= 32'sh8000_0000; maxi <= 4'd0;
    end else begin
      fc1_start <= 1'b0;
      fc2_start <= 1'b0;

      case (tstate)
        T_IDLE: begin
          // Reset state - all outputs are 0, wait for start button
          done <= 1'b0;
          predicted_digit <= 4'd0;
          tstate <= T_WAIT;
        end

        T_WAIT: begin
          // Wait for start button press
          done <= 1'b0;
          predicted_digit <= 4'd0;
          if (start_btn_edge) begin
            fc1_start <= 1'b1;
            tstate <= T_L1;
          end
        end

        T_L1: begin
          if (fc1_done) begin
            fc2_start <= 1'b1;
            tstate <= T_L2;
          end
        end

        T_L2: begin
          if (fc2_done) begin
            // init argmax
            idx  <= 4'd0;
            maxv <= y2_mem[0];
            maxi <= 4'd0;
            tstate <= T_ARGMAX;
          end
        end

        T_ARGMAX: begin
          if (idx < OUT2-1) begin
            idx <= idx + 1'b1;
            if (y2_mem[idx+1] > maxv) begin
              maxv <= y2_mem[idx+1];
              maxi <= idx + 1'b1;
            end
          end else begin
            predicted_digit <= maxi[3:0];
            tstate <= T_DONE;
          end
        end

        T_DONE: begin
          done <= 1'b1;
          // Stay in DONE state - predicted_digit remains displayed
          // Only reset will bring us back to IDLE
        end

        default: tstate <= T_IDLE;
      endcase
    end
  end

endmodule

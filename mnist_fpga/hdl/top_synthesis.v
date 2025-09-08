// hdl/top_synthesis.v - Modified for Vivado synthesis
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

  // Optional simple outputs for sim/board bring-up
  output reg        done,
  output reg [3:0]  predicted_digit  // fits 0..9
);

  wire rst = ~rst_n;

  // -------------------------
  // Memories (using BRAM inference for synthesis)
  // -------------------------
  // Input image (int8)
  (* ram_style = "block" *) reg signed [7:0] x_mem [0:IN1-1];
  
  // Layer1 params
  (* ram_style = "block" *) reg signed [7:0]  W1_mem [0:H1*IN1-1];
  (* ram_style = "block" *) reg signed [31:0] b1_mem [0:H1-1];

  // Layer1 output buffer (int8)
  (* ram_style = "distributed" *) reg signed [7:0]  h1_mem [0:H1-1];

  // Layer2 params
  (* ram_style = "block" *) reg signed [7:0]  W2_mem [0:OUT2*IN2-1];
  (* ram_style = "block" *) reg signed [31:0] b2_mem [0:OUT2-1];

  // Layer2 output (logits int32)
  (* ram_style = "distributed" *) reg signed [31:0] y2_mem [0:OUT2-1];

  // shift1 parameter (hardcoded for synthesis)
  localparam [5:0] shift1 = 6'd7;  // Update this value from shift1.txt

  // ------------- Memory Initialization for Synthesis -------------
  initial begin
    // Initialize memories with generated data
    // Note: In actual synthesis, these will be replaced by INIT attributes or COE files
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
    .clk(clk), .rst(rst), .start(fc1_start), .done(fc1_done),
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
    .clk(clk), .rst(rst), .start(fc2_start), .done(fc2_done),
    .x_addr(fc2_x_addr), .x_data(fc2_x_data),
    .w_addr(fc2_w_addr), .w_data(fc2_w_data),
    .b_addr(fc2_b_addr), .b_data(fc2_b_data),
    .y_we(fc2_y_we), .y_addr(fc2_y_addr), .y_data(fc2_y_data)
  );

  // -------------------------
  // Top-level FSM + Argmax
  // -------------------------
  localparam T_IDLE   = 3'd0;
  localparam T_L1     = 3'd1;
  localparam T_L2     = 3'd2;
  localparam T_ARGMAX = 3'd3;
  localparam T_DONE   = 3'd4;

  reg [2:0] tstate;

  // Argmax machinery
  reg [3:0]  idx;
  reg signed [31:0] maxv;
  reg [3:0]  maxi;

  always @(posedge clk) begin
    if (rst) begin
      tstate <= T_IDLE;
      fc1_start <= 1'b0;
      fc2_start <= 1'b0;
      done <= 1'b0;
      predicted_digit <= 4'd0;
      idx  <= 4'd0; maxv <= 32'sh8000_0000; maxi <= 4'd0;
    end else begin
      fc1_start <= 1'b0;
      fc2_start <= 1'b0;
      done      <= 1'b0;

      case (tstate)
        T_IDLE: begin
          // auto-start once after reset; in a real system you'd wait for a 'start' input
          fc1_start <= 1'b1;
          tstate    <= T_L1;
        end

        T_L1: begin
          if (fc1_done) begin
            fc2_start <= 1'b1;
            tstate    <= T_L2;
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
          done   <= 1'b1;   // pulse one cycle
          tstate <= T_IDLE; // or stay in DONE if you prefer
        end

        default: tstate <= T_IDLE;
      endcase
    end
  end

endmodule

// hdl/fc1_layer.v
`timescale 1ns/1ps
`include "util.vh"

// Serial fully-connected layer with right-shift + ReLU, int8 outputs.
// Reads:  x_mem (int8), W1 (int8, row-major), b1 (int32)
// Writes: h1_mem via write port (int8)
module fc1_layer #(
  parameter integer IN_DIM  = 784,
  parameter integer OUT_DIM = 32
)(
  input  wire                 clk,
  input  wire                 rst,
  input  wire                 start,
  output reg                  done,

  // shift to apply after accumulation (arith >>>)
  input  wire [5:0]           shift_right,

  // READ interfaces (behavioral arrays in top drive these)
  output reg  [$clog2(IN_DIM)-1:0]    x_addr,
  input  wire signed [7:0]            x_data,

  output reg  [$clog2(IN_DIM*OUT_DIM)-1:0] w_addr,
  input  wire signed [7:0]            w_data,

  output reg  [$clog2(OUT_DIM)-1:0]   b_addr,
  input  wire signed [31:0]           b_data,

  // WRITE port for hidden output (int8 after shift+ReLU)
  output reg                          y_we,
  output reg  [$clog2(OUT_DIM)-1:0]   y_addr,
  output reg  signed [7:0]            y_data
);

  localparam S_IDLE      = 3'd0;
  localparam S_LOAD_BIAS = 3'd1;
  localparam S_MAC       = 3'd2;
  localparam S_WRITE     = 3'd3;
  localparam S_DONE      = 3'd4;

  reg [2:0] state;

  reg [$clog2(OUT_DIM)-1:0] out_counter;  // neuron j
  reg [$clog2(IN_DIM)-1:0]  in_counter;   // input k

  reg signed [31:0] acc;
  reg signed [31:0] shr;
  reg signed [7:0]  q8;

  // address helpers
  wire [$clog2(IN_DIM*OUT_DIM)-1:0] w_flat_addr = out_counter*IN_DIM + in_counter;

  always @(posedge clk) begin
    if (rst) begin
      state <= S_IDLE; done <= 1'b0;
      out_counter <= 5'b0; in_counter <= 10'b0;
      acc <= 32'sd0;
      x_addr <= 10'b0; w_addr <= 15'b0; b_addr <= 5'b0;
      y_we <= 1'b0; y_addr <= 5'b0; y_data <= 8'sd0;
    end else begin
      y_we <= 1'b0;
      case (state)
        S_IDLE: begin
          done <= 1'b0;
          if (start) begin
            out_counter <= 5'b0;
            in_counter  <= 10'b0;
            b_addr      <= 5'b0;
            state       <= S_LOAD_BIAS;
          end
        end

        S_LOAD_BIAS: begin
          // init accumulator with bias of current output neuron
          b_addr <= out_counter;
          acc    <= b_data;
          // prepare first MAC addresses
          in_counter <= 10'b0;
          x_addr     <= 10'b0;
          w_addr     <= w_flat_addr;
          state      <= S_MAC;
        end

        S_MAC: begin
          // acc += w * x
          acc <= acc + $signed(w_data) * $signed(x_data);

          // next k
          if (in_counter == IN_DIM-1) begin
            state <= S_WRITE;
          end else begin
            in_counter <= in_counter + 1'b1;
            x_addr     <= in_counter + 1'b1;
            w_addr     <= out_counter*IN_DIM + (in_counter + 1'b1);
          end
        end

        S_WRITE: begin
          // shift, saturate, ReLU
          // arithmetic shift:
          // NOTE: >>> is arithmetic for signed in Verilog
          // clamp to int8 then ReLU (zero negatives)
          // Use helper for saturation
          shr = acc >>> shift_right;
          // Manual saturation instead of macro
          if (shr > 32'sd127)      q8 = 8'sd127;
          else if (shr < -32'sd128) q8 = -8'sd128;
          else                      q8 = shr[7:0];
          y_addr <= out_counter;
          y_data <= (q8[7]) ? 8'sd0 : q8;  // ReLU
          y_we   <= 1'b1;

          // next output neuron
          if (out_counter == OUT_DIM-1) begin
            state <= S_DONE;
          end else begin
            out_counter <= out_counter + 1'b1;
            state       <= S_LOAD_BIAS;
          end
        end

        S_DONE: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule

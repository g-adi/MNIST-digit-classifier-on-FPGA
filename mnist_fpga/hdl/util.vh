// hdl/util.vh
`ifndef UTIL_VH
`define UTIL_VH

// Saturate a signed 32-bit to signed 8-bit
function automatic [7:0] sat_int32_to_int8(input signed [31:0] x);
  begin
    if (x > 32'sd127)      sat_int32_to_int8 = 8'sd127;
    else if (x < -32'sd128) sat_int32_to_int8 = -8'sd128;
    else                    sat_int32_to_int8 = x[7:0];
  end
endfunction

`endif

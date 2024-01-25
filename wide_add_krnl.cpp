#include <ap_int.h>
#include <stdio.h>
#include <string.h>

#define BUFFER_SIZE 64
#define DATAWIDTH 512
#define VECTOR_SIZE (DATAWIDTH / 32) // vector size is 16 (512/32 = 16)
typedef ap_uint<DATAWIDTH> uint512_dt;

//TRIPCOUNT identifier
const unsigned int c_chunk_sz = BUFFER_SIZE;
const unsigned int c_size     = VECTOR_SIZE;

#define lm 4
#define ln 4
#define lp 4

#define m  1<<lm
#define n  1<<ln
#define p  1<<lp

/*
    Vector Addition Kernel Implementation using uint512_dt datatype
    Arguments:
        in1   (input)     --> Input Vector1
        in2   (input)     --> Input Vector2
        out   (output)    --> Output Vector
        size  (input)     --> Size of Vector in Integer
   */
extern "C"
{
    void wide_vadd(
        const uint512_dt *in1, // Read-Only Vector 1
        const uint512_dt *in2, // Read-Only Vector 2
        uint512_dt *out,       // Output Result
        int size               // Size in integer
    )
    {
#pragma HLS INTERFACE m_axi port = in1 max_write_burst_length = 32 max_read_burst_length = 32 offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = in2 max_read_burst_length = 32 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = out max_write_burst_length = 32 max_read_burst_length = 32 offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = in1 bundle = control
#pragma HLS INTERFACE s_axilite port = in2 bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control


    	const int loop = m;

    	int A_bram[n][m];
    	int B_bram[p][n];

    #pragma HLS ARRAY_PARTITION variable=A_bram cyclic factor 8 dim=2
    #pragma HLS ARRAY_PARTITION variable=B_bram cyclic factor 8 dim=1


    	for(int i = 0; i < m; i++){
    #pragma HLS loop_tripcount min=loop max=loop
    		uint512_dt in2local = in2[i];

			#pragma HLS PIPELINE II=1
    		for(int j = 0; j < m; j++){
    			B_bram[j][i] = in2local.range(32*(j + 1) - 1, j*32);
    		}
    	}


    	for(int i = 0; i < m; i++){
    	    #pragma HLS loop_tripcount min=loop max=loop

    	    		uint512_dt in1local = in1[i];
    	    		uint512_dt resultV;

    	    			for(int j = 0; j < m; j++){
    						#pragma HLS PIPELINE II=1
    	    				A_bram[i][j] = in1local.range(32*(j + 1) - 1, j*32);
    	    			}

    					for(int j = 0; j < m; j++){
    						#pragma HLS loop_tripcount min=loop max=loop

    						#pragma HLS PIPELINE II=1
    						int result = 0;
    						for(int k = 0 ; k < m ; k++){
    							result += A_bram[i][k] * B_bram[k][j] ;
    						}

    					resultV.range(32 * (j + 1) - 1, j * 32) = result;

    	    			}

    				out[i] = resultV;

    	}
    }
}
}


void testcase1() {
    int cachePower = 16; // cache size = 2^16B
    int blockPower = 5; // block size = 2^5B
    int stride = 1;
    int N = 1024;
    long[] A = new long[N];
    String cacheType = "DirectMapped";

    for (int i = 0;i < N;i+=1){
        A[i] = 0;
    }
}
// A -> 256

void testcase2() {
    int cachePower = 16; // cache size = 2^16B
    int blockPower = 5; // block size = 2^5B
    int N = 256;
    int[][] Z = new int[N][N];
    String cacheType = "DirectMapped";
    for (int i = 0; i < N; i += 1) {
        for (int j = 0; j < N; j += 1) {
            Z[i][j] = 0;
        }
    }
}
// Z -> 8192

void testcase3() {
    int cachePower = 18; // cache size = 2^18B
    int blockPower = 6; // block size = 2^6B
    int N = 256;
    int[][] A = new int[N][N];
    int[][] B = new int[N][N];
    int[][] C = new int[N][N];
    String cacheType = "DirectMapped";
    for (int i = 0; i < N; i += 1) {
        for (int j = 0; j < N; j += 1) {
            int sum = 0;
            for (int k = 0; k < N; k += 1) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}
// A -> 4096
// B -> 4096
// C -> 4096

void testcase4() {
    int s = 0;
    int cachePower = 18;
    int blockPower = 5;
    int setSize = 4;
    double[] A = new double[32768];
    int stride = 4;
    for (int it = 0; it < 10; it += 1) {
        for (int i = 0; i < 32768; i += stride) { 
            s += A[i];
        }
    }
}
// A -> 8192

void testcase5() {
    int cachePower = 17;
    int blockPower = 5;
    // String cacheType = "DirectMapped";
    String cacheType = "FullyAssociative";
    int N = 512;
    int[][] A = new int[N][N];
    int[][] B = new int[N][N];
    int[][] C = new int[N][N];
    for (int i = 0; i < N; i += 1) {
        for (int k = 0; k < N; k += 1) {
            for (int j = 0; j < N; j += 1) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
// C -> 32768
// A -> 32768
// B -> 16777216

void testcase6() {
    int cachePower = 17;
    int blockPower = 5;
    // String cacheType = "DirectMapped";
    String cacheType = "FullyAssociative";
    int N = 512;
    int[][] A = new int[N][N];
    int[][] B = new int[N][N];
    int[][] C = new int[N][N];
    for (int j = 0; j < N; j += 1) {
        for (int i = 0; i < N; i += 1) {
            for (int k = 0; k < N; k += 1) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
// C -> 32768
// A -> 16777216
// B -> 32768

void testcase7() {
    int cachePower = 24;
    int blockPower = 6;
    String cacheType = "DirectMapped";
    double[] y = new double[4096];
    double[][] X = new double[4096][4096];
    double[][] A = new double[4096][4096];
    for (int k = 0; k < 4096; k += 1) {
        for (int j = 0; j < 4096; j += 1) {
            for (int i = 0; i < 4096; i += 1) {
                y[i] = y[i] + A[i][j] * X[k][j];
            }
        }
    }
}
// y -> 512
// A -> 68719476736
// X -> 2097152

void testcase9() {
    int cachePower = 10;
    int blockPower = 5;
    long[][] A = new long[8][2];
    String cacheType = "DirectMapped";
    for(int i = 0; i<8; i+=1) {
        for(int j = 0; j<2; j+=1) {
            A[i][j] = 0;
        }
    }
}
// A -> 4

void testcase10() {
    int cachePower = 10;
    int blockPower = 2;
    long[][] A = new long[8][2];
    String cacheType = "DirectMapped";
    for(int i = 0; i<8; i+=1) {
        for(int j = 0; j<2; j+=1) {
            A[i][j] = 0;
        }
    }
}
// A -> 16

void testcase11() {
    int cachePower = 10;
    int blockPower = 5;

    long[][] A = new long[8][2];

    String cacheType = "DirectMapped";

    for(int i = 0; i<8; i+=2) {
        for(int j = 0; j<2; j+=1) {
            A[i][j] = 0;
        }
    }
}
// A -> 4

void testcase12() {
    int cachePower = 14;
    int blockPower = 5;
    String cacheType = "SetAssociative";
    int setSize = 8;
    int[][][] A = new int[64][64][64];
    int sum = 0;
    for(int j = 0; j < 64; j += 4){
        for(int k = 0; k < 64; k += 2) {
            for(int i = 0; i < 64; i += 8) {
                sum += A[k][i][j];
            }
        }
    }
}
// A -> 4096

void testcase13() {
    int cachePower = 20;
    int blockPower = 6;

    double[][][] X = new double[64][64][64];
    double[][][] A = new double[64][64][64];
    String cacheType = "DirectMapped";
    for(int k=0;k<64;k+=1){
        for(int j=0;j<64;j+=1){
            for(int i=0;i<64;i+=1){
                A[i][j][k] = 1 + X[k][j][i];
            }
        }
    }
}
// A -> 262144
// X -> 32768

void testcase14() {
    int cachePower = 14;
    int blockPower = 5;
    int stride = 1;
    int N = 64;
    int[][][] A = new int[N][N][N];
    int[][][] B = new int[N][N][N];
    int[][][] C = new int[N][N][N];

    String cacheType = "FullyAssociative";

    for (int j = 0;j < N;j+=1){
        for(int i=0;i<N;i+=stride){
            for(int k=0;k<N;k+=1){
                C[i][k][j] += A[i][j][k] * B[k][j][i];
            }
        }
    }
}
// A -> 32768
// B -> 32768
// C -> 262144

void testcase15() {
    int cachePower = 20;
    int blockPower = 6;

    double[][][] X = new double[64][64][64];
    double[][][] A = new double[64][64][64];
    String cacheType = "SetAssociative";
    int setSize = 8;
    for(int k=0;k<64;k+=1){
        for(int j=0;j<64;j+=1){
            for(int i=0;i<64;i+=1){
                A[i][j][k] = 1 + X[k][j][i];
            }
        }
    }
}
// A -> 262144
// X -> 32768


void testcase16() {
    int cachePower = 15;
    int blockPower = 5;
    int stride = 1;
    int N = 64;
    int[][][] A = new int[N][N][N];
    int[][][] B = new int[N][N][N];
    int[][][] C = new int[N][N][N];

    String cacheType = "SetAssociative";
    int setSize = 8;

    for (int j = 0;j < N;j+=1){
        for(int i=0;i<N;i+=stride){
            for(int k=0;k<N;k+=1){
                C[i][k][j] += A[i][j][k] * B[k][j][i];
            }
        }
    }
}
// A -> 32768
// B -> 262144
// C -> 262144

void testcase17() {
    int cachePower = 20;
    int blockPower = 6;

    double[][][] X = new double[64][64][64];
    double[][][] A = new double[64][64][64];
    String cacheType = "DirectMapped";

    for(int k=0;k<64;k+=2){
        for(int j=0;j<64;j+=4){
            for(int i=0;i<64;i+=1){
                A[i][j][k] = 1 + X[k][j][i];
            }
        }
    }
}
// A -> 32768
// X -> 4096

void testcase18() {
    int cachePower = 14;
    int blockPower = 5;
    int stride = 1;
    int N = 64;
    int[][][] A = new int[N][N][N];
    int[][][] B = new int[N][N][N];
    int[][][] C = new int[N][N][N];

    String cacheType = "FullyAssociative";

    for (int j = 0;j < N;j+=1){
        for(int i=0;i<N;i+=2){
            for(int k=0;k<N;k+=8){
                C[i][k][j] += A[i][j][k] * B[k][j][i];
            }
        }
    }
}
// A -> 16384
// B -> 4096
// C -> 2048

void testcase19() {
    int cachePower = 20;
    int blockPower = 6;

    double[][][] X = new double[64][64][64];
    double[][][] A = new double[64][64][64];
    String cacheType = "SetAssociative";
    int setSize = 8;

    for(int k=0;k<64;k+=2){
        for(int j=0;j<64;j+=4){
            for(int i=0;i<64;i+=8){
                A[i][j][k] = 1 + X[k][j][i];
            }
        }
    }
}
// A -> 1024
// X -> 4096

void testcase20() {
    int cachePower = 15;
    int blockPower = 5;
    int stride = 1;
    int N = 64;
    int[][][] A = new int[N][N][N];
    int[][][] B = new int[N][N][N];
    int[][][] C = new int[N][N][N];

    String cacheType = "SetAssociative";
    int setSize = 8;

    for (int j = 0;j < N;j+=2){
        for(int i=0;i<N;i+=2){
            for(int k=0;k<N;k+=2){
                C[i][k][j] += A[i][j][k] * B[k][j][i];
            }
        }
    }
}
// A -> 8192
// B -> 32768
// C -> 32768

void testcase21() {
    int cachePower = 15;
    int blockPower = 5;
    int stride = 1;
    int N = 64;
    int[][][] A = new int[N][N][8];
    int[][][] B = new int[N][N][8];
    int[][][] C = new int[N][N][8];

    String cacheType = "SetAssociative";
    int setSize = 8;

    for (int j = 0;j < N;j+=2){
        for(int i=0;i<N;i+=2){
            for(int k=0;k<8;k+=2){
                C[i][j][k] += A[j][i][k] * B[i][j][k];
            }
        }
    }
}
// A -> 1024
// B -> 1024
// C -> 1024

void testcase22() {
    int cachePower = 6; // cache size = 2^18B
    int blockPower = 4; // block size = 2^6B
    int N = 2;
    int[][][] A = new int[N][N][N];
    String cacheType = "DirectMapped";
    
    for (int i = 0; i < N; i += 1) {
        for (int k = 0; k < N; k += 1) {
            for (int j = 0; j < N; j += 1) {
                sum += A[i][k][j];
            }
        }
    }
}
// A -> 2

void testcase23() {
    int cachePower = 18; // cache size = 2^16B
    int blockPower = 5; // block size = 2^5B
    int stride = 1;
    int N = 32768;
    long[] A = new long[N];
    String cacheType = "SetAssociative";
    int setSize = 4;
    int stride = 16;
    int M = 256;
    for (int j=0;j<M;j+=1)
    {
        for(int i = 0;i < N;i+=stride){
            s += A[i];
        }
    }
}
// A -> 2048

void testcase24() {
    int cachePower = 21; // cache size = 2^16B
    int blockPower = 5; // block size = 2^5B
    int setSize = 2;
    // String cacheType = "SetAssociative";
    String cacheType = "DirectMapped";
    // String cacheType = "FullyAssociative";
    int N = 512;
    int sum = 0;
    int[][][] A = new int[8][N][N];
    for(int k = 0; k < N; k+=1) {
    	for(int j = 0; j < 8; j+=4) {
    		for(int i = 0; i < N; i+=1) {
    			sum = sum + A[j][i][k];
    		}
    	}
    }
}
// DirectMapped: A -> 524288
// SetAssociative: A -> 65536

void testcase24() {
    int cachePower = 21; // cache size = 2^16B
    int blockPower = 5; // block size = 2^5B
    int setSize = 2;
    String cacheType = "SetAssociative";
    // String cacheType = "DirectMapped";
    // String cacheType = "FullyAssociative";
    int N = 512;
    int sum = 0;
    int[][][] A = new int[8][N][N];
    for(int k = 0; k < N; k+=1) {
    	for(int j = 0; j < 8; j+=4) {
    		for(int i = 0; i < N; i+=1) {
    			sum = sum + A[j][i][k];
    		}
    	}
    }
}
// DirectMapped: A -> 524288
// SetAssociative: A -> 65536

void testcase25() {
    int cachePower = 24; // cache size = 2^16B
    int blockPower = 5; // block size = 2^5B
    int setSize = 2;
    // String cacheType = "SetAssociative";
    String cacheType = "DirectMapped";
    // String cacheType = "FullyAssociative";
    int N = 512;
    int sum = 0;
    int[][][] A = new int[8][N][N];
    for(int k = 0; k < N; k+=1) {
    	for(int j = 0; j < 2; j+=1) {
    		for(int i = 0; i < N; i+=1) {
    			sum = sum + A[j][i][k];
    		}
    	}
    }
}
// A -> 65536

void testcase26() {
    int cachePower = 10; // cache size = 2^16B
    int blockPower = 3; // block size = 2^5B
    int setSize = 2;
    // String cacheType = "SetAssociative";
    String cacheType = "DirectMapped";
    // String cacheType = "FullyAssociative";
    int N = 32;
    int sum = 0;
    int[][][] A = new int[2][N][4];
    
    for(int l = 0; l < 1000; l+=1){
        for(int j = 0; j < 2; j+=1) {
            for(int k = 0; k < 4; k+=1) {
                for(int i = 0; i < N; i+=1) {
                    sum = sum + A[j][i][k];
                }
            }
        }
    }
}
// A -> 128

void testcase27() {
    int cachePower = 10; // cache size = 2^16B
    int blockPower = 3; // block size = 2^5B
    int setSize = 2;
    // String cacheType = "SetAssociative";
    String cacheType = "DirectMapped";
    // String cacheType = "FullyAssociative";
    int N = 16;
    int sum = 0;
    int[][][] A = new int[4][N][8];
    
    for(int l =0;l<1000;l+=1){
        for(int j = 0; j < 4; j+=1) {
            for(int k = 0; k < 4; k+=1) {
                for(int i = 0; i < N; i+=1) {
                    sum = sum + A[j][i][k];
                }
            }
        }
    }
}
// A -> 128000

void testcase28() {
    int cachePower = 5; // cache size = 2^16B
    int blockPower = 4; // block size = 2^5B
    // String cacheType = "DirectMapped";
    String cacheType = "FullyAssociative";
    int sum = 0;
    int[][][] A = new int[4][2][1];
    for(int l =0;l<1000;l+=1){
        for(int j = 0; j < 4; j+=1) {
            for(int k = 0; k < 1; k+=1) {
                for(int i = 0; i < 1; i+=1) {
                    sum = sum + A[j][i][k];
                }
            }
        }
    }
}
// DirectMapped -> 2
// FullyAssociative -> 2

void testcase29() {
    int cachePower = 2; // cache size = 2^16B
    int blockPower = 1; // block size = 2^5B
    String cacheType = "DirectMapped";
    // String cacheType = "FullyAssociative";
    int sum = 0;
    int[][][] A = new int[1][1][1];
    for(int l =0;l<1000;l+=1){
        for(int j = 0; j < 1; j+=1) {
            for(int k = 0; k < 1; k+=1) {
                for(int i = 0; i < 1; i+=1) {
                    sum = sum + A[j][i][k];
                }
            }
        }
    }
}
// DirectMapped -> 2
// FullyAssociative -> 2

// // Matrix multiplication jik
// void hidden_testcase6(){
//     int cachePower = 16;
//     int blockPower = 5;
//     int N = 512;
//     int stride = 1;
//     int[][] A = new int[N][N];
//     int[][] B = new int[N][N];
//     int[][] C = new int[N][N];

//     String cacheType = "DirectMapped";
//     for (int j=0; j<N; j+=1) {
//         for (int i=0; i<N; i+=1) {
//             for (int k=0; k<N; k+=1){
//                 C[i][j] += A[i][k]*B[k][j];
//             }
//         }
//     }
// }
// // A=16777216
// // B=134217728
// // C=262144


// void hidden_testcase8(){
//     int cachePower = 20;
//     int blockPower = 6;
//     int N = 256;
//     int M = 128;
//     String cacheType = "SetAssociative";
//     int setSize = 4;
//     long sum = 0;
//     long[][] A = new long[N][M];

//     for (int it = 0; it < 16; it+=1) {
//         for (int i = 0; i < N; i += 16) {
//             for(int j = 0;j < M;j +=1){
//                 sum += A[i][j];
//             }

//         }
//     }
// }
// // A=256

// void hidden_testcase9(){
//     int cachePower = 18;
//     int blockPower = 5;
//     int N = 256;
//     String cacheType = "FullyAssociative";

//     int sum = 0;
//     int[][][] A = new int[N][N][N];
//     int[][][] B = new int[N][N][N];
//     int[][] C = new int[N][N];

//     for(int t = 0; t < 128;t+=1){
//         for (int i = 0; i < N; i +=1) {
//             for(int j = 0;j < N;j +=1){
//                 for(int k = 0;k < N;k+=1){
//                     A[i][j][k] = B[i][k][j] + C[k][i];
//                 }
//             }
//         }
//     }
// }
// // A=268435456
// // B=268435456
// // C=8192



#include "mex.h"
#include "math.h"
#include "matrix.h"
#include "stdlib.h"
#include "float.h"
#include "time.h"


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

double maxfun(double a, double b)
{
    if (a >= b) return a;
    else return b;
}

double minfun(double a, double b)
{
    if (a <= b) return a;
    else return b;
}

double uniform(double a, double b)
{
    return ((double) rand())/ RAND_MAX * (b -a) + a;
}

int binornd(double p)
{
    int x;
    double u;
    u = uniform(0.0, 1.0);
    x = (u <= p)? 1:0;
    return(x);
}


int getRandInt(int lowerLimit, int upperLimit){ // get an randomized interger in [lowerLimit, upperLimit]
    return lowerLimit + rand() % (upperLimit - lowerLimit + 1);
}

void randPerm(double *index, int N){
    int i, r1, r2, tmp;
    for(i=0; i < N; i++){
        r1 = getRandInt(0, N-1);
        r2 = getRandInt(0, N-1);
        if (r1!=r2){
            tmp =  index[r1];
            index[r1]= index[r2];
            index[r2] = tmp;
        }
    }
}

double squareNorm(double *x, int len){
    int i;
    double sum = 0;
    for(i = 0;i < len; i++){
        sum = sum + x[i] * x[i];
    }
    return sum;
}

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    if(nrhs != 5) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "5 inputs required.");
    }
    if(mxIsSparse(prhs[0])==0 && mxIsSparse(prhs[1])==0){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "data/label matrix is not sparse!");
    }
    
    
    double *data, *labels, *w, *index, *x, *y, C, minV, maxV, f_t_1, f_t_2, alpha, beta, snx;
    int i,j,k,p,N,d,L,low,high,nonzerosNum,low1,high1,epoch,o,r_t,s_t,maxIterNum,iter;
    mwIndex *ir, *jc, *ir1, *jc1;
    int * idx;
    /*Read Input Data*/
    data = mxGetPr(prhs[0]);  // use the mxGetPr function to point to the input matrix data.
    labels = mxGetPr(prhs[1]);
    index = mxGetPr(prhs[2]);
    epoch = mxGetScalar(prhs[3]);
    C = mxGetScalar(prhs[4]);
    
    // a column is an instance
    d = (int)mxGetM(prhs[0]); //get Number of rows in array
    N = (int)mxGetN(prhs[0]); //get Number of columns in array
    L = (int)mxGetM(prhs[1]); //the dimension of each label vector
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    ir1 = mxGetIr(prhs[1]);
    jc1 = mxGetJc(prhs[1]);
    
    /* preparing outputs */
    plhs[0] = mxCreateDoubleMatrix(d, L+1, mxREAL);
    w = mxGetPr(plhs[0]);
    
    // plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    // errNum = mxGetPr(plhs[1]);
    
    double * pred_v = Malloc(double,L+1);
    srand(0);
    
    for (o = 1; o <= epoch; o++){
        if (o > 1) randPerm(index, N);
        /* start loop */
        for(i = 0; i < N; i++)
        {
            j = index[i] - 1;
            
            // get each instance
            low = jc[j]; high = jc[j+1];
            nonzerosNum = high - low;
            x = Malloc(double,nonzerosNum);
            idx = Malloc(int,nonzerosNum); // the indices of the non-zero values in x
            for (k = low; k < high; k++){
                x[k-low] = data[k];
                idx[k-low] = ir[k];
            }
            
            // get each label vector
            y = Malloc(double,L);
            for (k = 0; k < L; k++){
                y[k] = -1;
            }
            low1 = jc1[j]; high1 = jc1[j+1];
            for (k = low1; k < high1; k++){
                y[ir1[k]] = labels[k];
            }
            
            if ((high1 - low1)> 0 && (high1 - low1) < L)
                maxIterNum = (high1 - low1) * (L - (high1 - low1));
            else
                maxIterNum = L;
            
            for (iter = 0; iter < maxIterNum; iter++){
                // get predicted value for each label
                if (iter == 0){ // re-compute each predicted value
                    for (k = 0; k <= L; k++){
                        pred_v[k] = 0;
                        for (p = 0; p < nonzerosNum; p++){
                            pred_v[k] += w[k*d + idx[p]] * x[p];
                        }
                    }
                }else{ // re-compute the predicted value for label r_t, s_t, L
                    pred_v[r_t] = 0;
                    pred_v[s_t] = 0;
                    pred_v[L] = 0;
                    for (p = 0; p < nonzerosNum; p++){
                        pred_v[r_t] += w[r_t*d + idx[p]] * x[p];
                        pred_v[s_t] += w[s_t*d + idx[p]] * x[p];
                        pred_v[L] += w[L*d + idx[p]] * x[p];
                    }
                }
                
                // compute r_t and s_t
                minV = DBL_MAX;
                maxV = -DBL_MAX;
                r_t = -1;
                s_t = -1;
                for(k = 0; k < L; k++){
                    if(y[k] == 1){
                        if (pred_v[k] < minV){
                            minV = pred_v[k];
                            r_t = k;
                        }
                    }
                    else
                    {
                        if(pred_v[k] > maxV){
                            maxV = pred_v[k];
                            s_t = k;
                        }
                    }
                }
                if (r_t != -1)
                    f_t_1 = 1 - (pred_v[r_t] - pred_v[L]);
                else{
                    f_t_1 = NAN;
                    r_t = 0;
                }
                if (s_t != -1)
                    f_t_2 = 1 - (pred_v[L] - pred_v[s_t]);
                else{
                    f_t_2 =  NAN;
                    s_t = 0;
                }
                
                if(iter == 0){
                    snx = squareNorm(x,nonzerosNum);
                    if (snx == 0)    break;
                }
                // compute alpha and beta
                if (isnan(f_t_1)!= 0){
                    alpha = 0;
                    beta = minfun(maxfun(0, f_t_2 / (2 * snx)), C);
                }else if (isnan(f_t_2)!=0){
                    alpha = minfun(maxfun(0, f_t_1 / (2 * snx)), C);
                    beta = 0;
                }else if(f_t_1 <=0 && f_t_2<=0){
                    alpha = 0;
                    beta = 0;
                }else if(f_t_1 <= -0.5 * f_t_2 && f_t_2>0 && f_t_2 <= 2*C*snx){
                    alpha = 0;
                    beta = f_t_2 / (2 * snx);
                }else if(f_t_1 <= -C*snx && f_t_2 > 2*C*snx ){
                    alpha = 0;
                    beta = C;
                }else if(f_t_1 > -C*snx && f_t_1 <= C*snx && f_t_2 >= -0.5*f_t_1 + 3*C*snx/2){
                    alpha = 0.5 * (C + f_t_1/snx);
                    beta = C;
                }else if(f_t_1 > C*snx && f_t_2 > C*snx){
                    alpha = C;
                    beta = C;
                }else if(f_t_2 > -C*snx && f_t_2 <= C*snx && f_t_1 >= -0.5*f_t_2 + 3*C*snx/2){
                    alpha = C;
                    beta = 0.5 * (C + f_t_2/snx);
                }else if(f_t_2 <= -C*snx && f_t_1 > 2*C*snx){
                    alpha = C;
                    beta = 0;
                }else if(f_t_2 <= -0.5 * f_t_1 && f_t_1 > 0 && f_t_1 <= 2*C*snx){
                    alpha = f_t_1 / (2 * snx);;
                    beta = 0;
                }else{
                    alpha = (2*f_t_1 + f_t_2) / (3 * snx);
                    beta = (f_t_1 + 2*f_t_2) / (3 * snx);
                }
//             printf("alpha = %f,beta = %f \n\n", alpha, beta);
                if(alpha == 0 && beta ==0)  break;
                // update the model
                for (p = 0; p < nonzerosNum; p++){
                    w[r_t*d + idx[p]] =  w[r_t*d + idx[p]] + alpha * x[p];
                    w[s_t*d + idx[p]] =  w[s_t*d + idx[p]] - beta * x[p];
                    w[L*d + idx[p]] =  w[L*d + idx[p]] - (alpha - beta) * x[p];
                }
            }
//             printf("iter = %d \n", iter);
            free(x);
            free(idx);
            free(y);
        }
    }
    
    free(pred_v);
}


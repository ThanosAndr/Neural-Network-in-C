/*
 ============================================================================
 Name        : Net.c
 Author      : Thanos
 Version     :
 Copyright   : 
 Description : Hello World in C, Ansi-style
 ============================================================================
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
/*Neural Networks Defines */
#define NUMPAT 1589
#define TESTNUMPAT 1297
#define NUMIN  12
#define NUMHID 10
#define NUMOUT 3



#define rando() ((double)rand()/((double)RAND_MAX+1))
double sigmoid(double x);
void read_data(int sizein,int sizeout, int sizepat,int testsizepat,double Input[sizepat][sizein],int Target[sizepat][sizeout],double TestInput[testsizepat][sizein],int TestTarget[testsizepat][sizeout]);
void count_target(int sizeout,int sizepat,int Target[sizepat][sizeout],int n[sizeout]);
void initialize_weights(int sizein,int sizeout,int sizehid,double smallwt,double WeightIH[sizein][sizehid],double WeightHO[sizehid][sizeout],double** DeltaWeightIH,double** DeltaWeightHO);
void train_hidden(int sizein,int sizeout,int sizepat, int sizehid,int epochs,double eta,double alpha,int n[sizeout],double Input[sizepat][sizein],double** Hidden,double** Output ,int Target[sizepat][sizeout],double WeightIH[sizein][sizehid],double WeightHO[sizehid][sizeout],double** DeltaWeightIH,double** DeltaWeightHO);
void print_results(int sizein,int sizeout,int sizepat,int n[sizeout],double** Output,int Target[sizepat][sizeout]);
void forward_test(int sizein,int sizeout,int sizepat,int sizehid,double in[sizepat][sizein],int out[sizepat][sizeout],int n1[sizeout],double WeightIH[sizein][sizehid],double WeightHO[sizehid][sizeout]);
void exportnn(int sizein,int sizeout,int sizehid,int sizepat,double WeightIH[sizein][sizehid],double WeightHO[sizehid][sizeout]);

int main(void) {
	int epochs = 80;
	double  eta = 0.0001, alpha = 0.9, smallwt = 0.5; // eta : 0.001 , alpha = 0.9
	/* Declare Neural Network Variables */
	double **DeltaWeightIH;//[NUMIN][NUMHID]
	double **DeltaWeightHO;//[NUMHID][NUMOUT];
	double WeightIH[NUMIN][NUMHID];
	double **Hidden;//[NUMPAT][NUMHID];
	double WeightHO[NUMHID][NUMOUT];
	double **Output;

	/* Training Data */
	double Input[NUMPAT][NUMIN];
	int Target[NUMPAT][NUMOUT];
	int n[NUMOUT]= {0,0,0};
	/* Testing Data */
	double TestInput[TESTNUMPAT][NUMIN];
	int TestTarget[TESTNUMPAT][NUMOUT];
	int n1[NUMOUT]={0,0,0};
	int    NumPattern = NUMPAT, NumInput = NUMIN, NumHidden = NUMHID, NumOutput = NUMOUT,NumTestPattern = TESTNUMPAT;

	int i;
	Hidden = (double **) malloc(NUMPAT*sizeof(double*));
	for(i=0;i<NUMPAT;i++){
		Hidden[i] = (double *) malloc(NUMHID*sizeof(double));
	}
	DeltaWeightIH = (double **) malloc(NUMIN*sizeof(double*));
	for(i=0;i<NUMIN;i++){
		DeltaWeightIH[i] = (double *) malloc(NUMHID*sizeof(double));
	}
	DeltaWeightHO = (double **) malloc(NUMHID*sizeof(double*));
	for(i=0;i<NUMHID;i++){
		DeltaWeightHO[i] = (double *) malloc(NUMOUT*sizeof(double));
	}
	Output = (double **) malloc(NUMPAT*sizeof(double*));
	for(i=0;i<NUMPAT;i++){
		Output[i] = (double *) malloc(NUMOUT*sizeof(double));
	}
	//---------------------------------------------------------------------------------------//

	/* Read the Datasets for Training & Testing from .txt Files */
	read_data(NumInput,NumOutput,NumPattern,NumTestPattern,Input,Target,TestInput,TestTarget);
	//---------------------------------------------------------------------------------------//

	/* Count the Number of Samples we Have Per User for Training & Testing Data */
	count_target(NumOutput,NumPattern,Target,n);
	count_target(NumOutput,NumTestPattern,TestTarget,n1);
	//---------------------------------------------------------------------------------------//

	/* Initialize the Neural Networks Weights */
	initialize_weights(NumInput,NumOutput,NumHidden,smallwt,WeightIH,WeightHO,DeltaWeightIH,DeltaWeightHO);
	//---------------------------------------------------------------------------------------//

	/* Initialize the Hidden Unit & Print the Training Results */
	train_hidden(NumInput,NumOutput,NumPattern,NumHidden,epochs,eta,alpha,n,Input,Hidden,Output,Target,WeightIH,WeightHO,DeltaWeightIH,DeltaWeightHO);
	//---------------------------------------------------------------------------------------//
	free(Hidden);free(DeltaWeightHO);free(DeltaWeightIH);free(Output);

	/* Forward Propagation of Neural Network with Test Dataset & Print Testing Results */
	forward_test(NumInput,NumOutput,NumTestPattern,NumHidden,TestInput,TestTarget,n1,WeightIH,WeightHO);
	//---------------------------------------------------------------------------------------//

	/* Export the Weights of the Neural Network to .txt files */
	exportnn(NumInput,NumOutput,NumHidden,NumPattern,WeightIH,WeightHO);
	//---------------------------------------------------------------------------------------//

	return 1;
}
/* Read the Datasets for Training & Testing from .txt Files */
void read_data(
		int sizein,
		int sizeout,
		int sizepat,
		int testsizepat,
		double Input[sizepat][sizein],
		int Target[sizepat][sizeout],
		double TestInput[testsizepat][sizein],
		int TestTarget[testsizepat][sizeout]){

	int i, j;
	FILE * itrfp = fopen("Data/Input.txt","r");
	FILE * ttrfp = fopen("Data/Target.txt","r");
	FILE * itefp = fopen("Data/TestInput.txt","r");
	FILE * ttefp = fopen("Data/TestTarget.txt","r");

	for (i=0; i <sizepat; i++){
		for (j=0;j<sizein;j++){
			fscanf(itrfp,"%lf",&Input[i][j]);
		}
	}
	for (i=0; i <sizepat; i++){
		for (j=0;j<sizeout;j++){
			fscanf(ttrfp,"%d",&Target[i][j]);

		}
	}
	for (i=0; i <testsizepat; i++){
		for (j=0;j<sizein;j++){
			fscanf(itefp,"%lf",&TestInput[i][j]);
		}
	}
	for (i=0; i <testsizepat; i++){
		for (j=0;j<sizeout;j++){
			fscanf(ttefp,"%d",&TestTarget[i][j]);
		}
	}
}
/* Count the Number of Samples we Have Per User for Training & Testing Data */
void count_target(
		int sizeout,
		int sizepat,
		int Target[sizepat][sizeout],
		int n[sizeout]){
	int i,j;
	for (i=0;i<sizepat;i++){
		for (j=0;j<sizeout;j++){
			if (j==0 && Target[i][j]==1){
				n[j]++;
			}
			if (j==1 && Target[i][j]==1){
				n[j]++;
			}
			if (j==2 && Target[i][j]==1){
				n[j]++;
			}
		}
	}
//			fprintf(stdout,"%d\t%d\t%d\n\n",n[0],n[1],n[2]);  //Used only for debuging

}
/* Initialize the Neural Networks Weights */
void initialize_weights(
		int sizein,
		int sizeout,
		int sizehid,
		double smallwt,
		double WeightIH[sizein][sizehid],
		double WeightHO[sizehid][sizeout],
		double **DeltaWeightIH,
		double **DeltaWeightHO){
	int    i, j, k;
	for( j = 0 ; j < sizehid ; j++ ) {    /* initialize WeightIH and DeltaWeightIH */
		for( i = 0 ; i < sizein ; i++ ) {
			DeltaWeightIH[i][j] = 0.0 ;
			WeightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
		}
	}
	for( k = 0 ; k < sizeout ; k ++ ) {    /* initialize WeightHO and DeltaWeightHO */
		for( j = 0 ; j < sizehid ; j++ ) {
			DeltaWeightHO[j][k] = 0.0 ;
			WeightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
		}
	}
}
/* Train The Neural Network */
void train_hidden(
		int sizein,
		int sizeout,
		int sizepat,
		int sizehid,
		int epochs,
		double eta,
		double alpha,
		int n[sizeout],
		double Input[sizepat][sizein],
		double **Hidden,
		double **Output,
		int Target[sizepat][sizeout],
		double WeightIH[sizein][sizehid],
		double WeightHO[sizehid][sizeout],
		double **DeltaWeightIH,
		double **DeltaWeightHO){

	double Error,SumO[sizepat][sizeout],SumH[sizepat][sizehid];//ranpat[sizepat];

	int    epoch,i, j, k, p, np, op;
	double *DeltaO;
	DeltaO = (double*)malloc(sizeout*sizeof(double));
	double *SumDOW;
	SumDOW = (double*)malloc(sizehid*sizeof(double));
	double *DeltaH;
	DeltaH = (double*)malloc(sizehid*sizeof(double));
	int *ranpat;
	ranpat = (int*)malloc(sizepat*sizeof(int));

	srand(time(NULL));   // should only be called once



	for( epoch = 0 ; epoch < epochs ; epoch++) {    /* iterate weight updates */
		for( p = 0 ; p < sizepat ; p++ ) {    /* randomize order of individuals */
			ranpat[p] = p ;
		}
		for( p = 0 ; p < sizepat ; p++) {
			np = p + rando() * ( sizepat + 1 - p ) ;
			op = ranpat[p];
			ranpat[p] = ranpat[np];
			ranpat[np] = op ;

		}
		Error = 0.0 ;
		for( np = 0 ; np < sizepat ; np++ ) {    /* repeat for all the training patterns */
			p = ranpat[np];
			//fprintf(stdout,"\n%d\t\n",p);
			for( j = 0 ; j < sizehid ; j++ ) {    /* compute hidden unit activations */

				SumH[p][j] = WeightIH[0][j];
				for( i = 0 ; i < sizein ; i++ ) {
					SumH[p][j] += Input[p][i] * WeightIH[i][j] ;

				}
				Hidden[p][j] = sigmoid(SumH[p][j]) ;
			}
			for( k = 0 ; k < sizeout ; k++ ) {    /* compute output unit activations and errors */
				SumO[p][k] = WeightHO[0][k] ;
				for( j = 0 ; j < sizehid ; j++ ) {
					SumO[p][k] += Hidden[p][j] * WeightHO[j][k] ;
				}
				Output[p][k] = sigmoid(SumO[p][k]);   /* Sigmoidal Outputs */

				//				Error += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]) ;   /* SSE */
				              Error -= ( Target[p][k] * log( Output[p][k] ) + ( 1.0 -  Target[p][k] ) * log( 1.0 - Output[p][k] ) ) ;   /*  Cross-Entropy Error */
				//             	  DeltaO[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;   /* Sigmoidal Outputs, SSE */
				              DeltaO[k] =Target[p][k] - Output[p][k];   /*  Sigmoidal Outputs, Cross-Entropy Error */
				            //  fprintf(stdout, "%d\t%f\t%d\t%f\n",p, DeltaO[k],Target[p][k],Output[p][k]);
			}
//			fprintf(stdout, "\n\n");
			/* Initialize the Hidden Unit: End */
			//---------------------------------------------------------------------------------------//
			/*  Used Only for Training */
			/* Back Propagation of Errors to the Hidden Layer: Start */
			for( j = 0 ; j < sizehid ; j++ ) {    /* 'Back Propagate' errors to hidden layer */
				SumDOW[j] = 0.0 ;
				for( k = 0 ; k < sizeout ; k++ ) {
					SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;
				}
				DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
			}
			for( j = 0 ; j < sizehid ; j++ ) {     /* Update WeightIH */
				DeltaWeightIH[0][j] = eta * DeltaH[j] + alpha * DeltaWeightIH[0][j] ;
				WeightIH[0][j] += DeltaWeightIH[0][j] ;
				for( i = 0 ; i < sizein ; i++ ) {
					DeltaWeightIH[i][j] = eta * Input[p][i] * DeltaH[j] + alpha * DeltaWeightIH[i][j];
					WeightIH[i][j] += DeltaWeightIH[i][j] ;
				}
			}

			for( k = 0 ; k < sizeout ; k ++ ) {    /* Update WeightHO */
				DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k] ;
				WeightHO[0][k] += DeltaWeightHO[0][k] ;
				for( j = 0 ; j < sizehid ; j++ ) {
					DeltaWeightHO[j][k] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k] ;
					WeightHO[j][k] += DeltaWeightHO[j][k] ;
				}
			}
			/* Back Propagation of Errors to the Hidden Layer: End */
			/* Used Only for Training */
			//---------------------------------------------------------------------------------------//
		}
		/* Printing the Error Percentage every 100 epochs: Start */
		if( epoch%10 == 0 ){ fprintf(stdout, "\nEpoch %-5d :   Error = %f \n", epoch, Error) ;}
		/* Printing the Error Percentage every 100 epochs: Start */
	}	fprintf(stdout,"\n\n");
	//	for (k = 0; k < NumOutput; k++) {
	//		fprintf(stdout, "Target%-4d\tOutput%-4d\t", k, k);
	//	}
	fprintf(stdout,"Training Results: \n");
	print_results(sizein,sizeout,sizepat,n,Output,Target);
	free(DeltaO);free(SumDOW);free(DeltaH);free(ranpat);
}
/* Print The Results of Training & Testing */
void print_results(int sizein,
		int sizeout,
		int sizepat,
		int n[sizeout],
		double** Output,
		int Target[sizepat][sizeout]){
	int p,k;
	double SR[sizeout];
	double maxrow[sizepat], temp;
	SR[0] = 0;
	SR[1] = 0;
	SR[2] = 0;
	//SR[3] = 0;
	for (p = 0; p < sizepat; p++) {
		maxrow[p] = Output[p][0];
		for (k = 0; k < sizeout; k++) {
			temp = Output[p][k];
			if (maxrow[p] < temp) {
				maxrow[p] = temp;
			}
		}
	}
	for (p = 0; p < sizepat; p++) {
	//	fprintf(stdout, "\n%d\t", p);

		/* Calculate the Successful Outputs */
		for (k = 0; k < sizeout; k++) {
		//	fprintf(stdout, "%d\t%f\t", Target[p][k], Output[p][k]);

			if (Target[p][k] == 1) {
				if (Output[p][k] == maxrow[p]) {
					SR[k] = SR[k] + 1;
				}
			}

		}
		//fprintf(stdout, "%f\t", maxrow[p]) ;
	}
	/* Calulate the Success Rate of each User */

	for (k = 0; k < sizeout; k++) {
		//		if (SR[k] != 0){
		fprintf(stdout, "\n\nSuccess Rate of User %i = %lf\t", k,(SR[k] / n[k]) * 100);
		fprintf(stdout, "%f\t", SR[k]) ;
		//		}
	}
	//	fprintf(stdout, "\n\n");
}
/* Forward Propagation for Testing */
void forward_test(
		int sizein,
		int sizeout,
		int sizepat,
		int sizehid,
		double in[sizepat][sizein],
		int out[sizepat][sizeout],
		int n1[sizeout],
		double WeightIH[sizein][sizehid],
		double WeightHO[sizehid][sizeout]){

	int i,j,p;
	double **tout;
	tout = (double **) malloc(sizepat*sizeof(double*));
	for(i=0;i<sizepat;i++){
		tout[i] = (double *) malloc(sizeout*sizeof(double));
	}
	double** sumIH;
	sumIH = (double **) malloc(sizepat*sizeof(double*));
	for(i=0;i<sizepat;i++){
		sumIH[i] = (double *) malloc(sizehid*sizeof(double));
	}
	double** sumHO;
	sumHO = (double **) malloc(sizepat*sizeof(double*));
	for(i=0;i<sizepat;i++){
		sumHO[i] = (double *) malloc(sizeout*sizeof(double));
	}
	// Input Layer -> Hidden Layer
	for(p=0;p<sizepat;p++){
		for(i=0;i<sizehid;i++){
			sumIH[p][i] = WeightHO[0][i] ;
			for(j=0;j<sizein;j++){
				sumIH[p][i] += WeightIH[j][i]*in[p][j];

			}
			// sumh1[i]
			sumIH[p][i] = sigmoid(sumIH[p][i]);

		}

		// Hidden Layer -> Output Layer
		for(j=0;j<sizeout;j++){
			sumHO[p][j] = WeightHO[0][j];
			for(i=0;i<sizehid;i++){
				sumHO[p][j]+=sumIH[p][i]*WeightHO[i][j];
			}
			// sumh2[i]

			tout[p][j] = sigmoid(sumHO[p][j]);
			//						fprintf(stdout,"%lf\r",tout[p][j]);

		}
		//				fprintf(stdout,"\n");
	}
	fprintf(stdout,"\nTesting Results: \n");
	free(sumIH);free(sumHO);
	print_results(sizein,sizeout,sizepat,n1,tout,out);
	free(tout);


}
/* Export the Weights of the Neural Network to .txt files */
void exportnn(
		int sizein,
		int sizeout,
		int sizehid,
		int sizepat,
		double WeightIH[sizein][sizehid],
		double WeightHO[sizehid][sizeout]){
	int a, b;
	FILE *f1 = fopen("Weights/WeightIH.txt", "w");
	for (a = 0; a < sizein; a++) {
		for (b = 0; b < sizehid; b++) {
			fprintf(f1, "%lf\t", WeightIH[a][b]);
		}
		fprintf(f1, "\n");
	}

	fclose(f1);

	FILE *f2 = fopen("Weights/WeightHO.txt", "w");
	for (a = 0; a < sizehid; a++) {
		for (b = 0; b < sizeout; b++) {
			fprintf(f2, "%lf\t", WeightHO[a][b]);
		}
		fprintf(f2, "\n");
	}
	fclose(f2);
}
/* Activation Function */
double sigmoid(double x){
	return 1.0/(1.0 + exp(-1.0*x));
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
//#include <time.h>

#define DATA_SIZE 10000 // maximum size of data points
#define DIMENSIONS 100 // maximum size of data dimensions


void FindGradient(double data[DATA_SIZE][DIMENSIONS], int sizeOfData, int dimensions, double sigma, double x[DIMENSIONS], double gradient[DIMENSIONS])
// finds the gradient of a point x according to QC
// data - an array of size [DATA_SIZE][DIMENSIONS] with data points
// sizeOfData - actual number of data points (must be <= than DATA_SIZE)
// dimensions - actual number of dimensions (must be <= than DIMENSIONS)
// sigma - Gaussian width in QC
// x - a single point, where the gradient will be calculated
// gradient - the gradient
{
	double qInv = 2 * sigma*sigma;
	double q = 1 / qInv;
	double difference[DATA_SIZE][DIMENSIONS];
	double squaredDifference[DATA_SIZE];
	double gaussian[DATA_SIZE];
	double laplacian = 0;
	double parzen = 0;
	double potential;
	double temp;

	int i;
	int j;

	for (j = 0; j < dimensions; j++)
	{
		gradient[j] = 0;
	}

	for (i = 0; i < sizeOfData; i++)
	{
		squaredDifference[i] = 0;
		for (j = 0; j < dimensions; j++)
		{
			difference[i][j] = x[j] - data[i][j];
			squaredDifference[i] += (difference[i][j])*(difference[i][j]);
		}
		gaussian[i] = exp(-q*squaredDifference[i]);
		laplacian += gaussian[i] * squaredDifference[i];
		parzen += gaussian[i];
	}
	potential = laplacian/parzen+qInv;
	for (i = 0; i < sizeOfData; i++)
	{
		temp = gaussian[i] * (potential - squaredDifference[i]);
		for (j = 0; j < dimensions; j++)
		{
			gradient[j] += temp*difference[i][j];
		}
	}
}


double* PerformGradientDescent(double data[DATA_SIZE][DIMENSIONS], int sizeOfData, int dimensions, double sigma, int xInd, double stepSize, int iterations, int normalizeData)
// performs GD steps
// data - an array of size [DATA_SIZE][DIMENSIONS] with data points
// sizeOfData - actual number of data points (must be <= than DATA_SIZE)
// dimensions - actual number of dimensions (must be <= than DIMENSIONS)
// sigma - Gaussian width in QC
// xInd - the point that will be undergo GD double is data[xInd]
// stepSize - size of each GD step
// iterations - number of GD iterations
// normalizeData - (boolean) should normalize the data each step?
// output: double array of size [DIMENSIONS] with final position of point after GD (memory for the array allocated inside this function!)
{
	double normalizedData[DATA_SIZE][DIMENSIONS];
	double* x = malloc(sizeof(double)*DIMENSIONS);
	double gradient[DIMENSIONS];
	double norm;
	int i;
	int j;

	if (normalizeData)
	{
		// perform normalization on data
		for (i = 0; i < sizeOfData; i++)
		{
			norm = 0;
			for (j = 0; j < dimensions; j++)
			{
				norm += (data[i][j])*(data[i][j]);
			}
			if (norm > 0)
			{
				norm = sqrt(norm);
				for (j = 0; j < dimensions; j++)
				{
					normalizedData[i][j] = data[i][j] / norm;
				}
			}
			else
			{
				normalizedData[i][j] = 0;
			}
		}
	}
	else
	{
		//  copy data without normalization
		for (i = 0; i < sizeOfData; i++)
		{
			for (j = 0; j < dimensions; j++)
			{
				normalizedData[i][j] = data[i][j];
			}
		}
	}

	// copy point data[xInd] to x
	for (j = 0; j < dimensions; j++)
	{
		x[j] = normalizedData[xInd][j];
	}

	// loop on iterations of gradient descent
	for (i = 0; i < iterations; i++)
	{
		// compute gradient
		FindGradient(normalizedData, sizeOfData, dimensions, sigma, x, gradient);

		// normalize gradient
		norm = 0;
		for (j = 0; j < dimensions; j++)
		{
			norm += (gradient[j])*(gradient[j]);
		}
		if (norm > 0)
		{
			norm = sqrt(norm);
			for (j = 0; j < dimensions; j++)
			{
				gradient[j] = gradient[j] / norm;
			}
		}

		// apply gradient
		for (j = 0; j < dimensions; j++)
		{
			x[j] -= stepSize*gradient[j];
		}

		// normalize x, if needed
		if (normalizeData)
		{
			norm = 0;
			for (j = 0; j < dimensions; j++)
			{
				norm += (x[j])*(x[j]);
			}
			if (norm > 0)
			{
				norm = sqrt(norm);
				for (j = 0; j < dimensions; j++)
				{
					x[j] = x[j] / norm;
				}
			}
		}
	}

	return x;
}


int main()
{
    double sigma = 8; // Gaussian width in QC
	double stepSize = 1; // step size of GD
	int iterations = 50; //number of iterations of GD
	int normalizeData = 0; // (boolean) should normalize data every GD step?

	//time_t t = time(NULL);

	double data[DATA_SIZE][DIMENSIONS]; // data points which are going to be clustered

	char* inFilename = "in.csv"; //data will be read from this file (comma-seperated)
	char* outFilename = "out.csv"; // results will be written into this file (comma-seperated)
	FILE* file = fopen(inFilename,"r");

	int sizeOfData = 0; // number of data points which will be clustered. This number is calculated later
	int dimensions = 0; // dimension of data points which will be clustered. This number is calculated later

	char* record;
	char* line;
	char buffer[DIMENSIONS*30];

    // read data from file
	while ((line = fgets(buffer, sizeof(buffer), file)) != NULL)
	{
		dimensions = 0;
		record = strtok(line, ",");
		while (record != NULL)
		{
			data[sizeOfData][dimensions] = strtod(record,NULL);
			dimensions++;
			record = strtok(NULL, ",");
		}
		sizeOfData++;
	}
	fclose(file);


	int i;
	int j;
	int c;

	double** x = malloc(sizeof(double*)*sizeOfData); // x will have results of clustering

    //printf("prepreocessing time: %.3f\n",difftime(time(NULL),t));
    //t = time(NULL);

    for (i = 0; i < sizeOfData ; i++)
	{
		x[i] = PerformGradientDescent(data, sizeOfData, dimensions, sigma, i, stepSize, iterations, normalizeData);
	}

	//printf("GD time: %.3f\n",difftime(time(NULL),t));
    //t = time(NULL);

	// collect results from threads

    // write to file
	file = fopen(outFilename, "w");
	for (i = 0; i < sizeOfData; i++)
	{
		for (j = 0; j < (dimensions-1); j++)
		{
			fprintf(file, "%f,",x[i][j]);
		}
		fprintf(file, "%f\n", x[i][j]);
	}
	fclose(file);

	//printf("postprocessing time: %.3f\n",difftime(time(NULL),t));

	system("PAUSE");
	return 0;
}

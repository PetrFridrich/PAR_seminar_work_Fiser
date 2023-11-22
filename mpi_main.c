#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define SIGNAL_LENGTH 100000
#define ITERATIONS 50
#define REPEAT 100
#define DATA_LEN (SIGNAL_LENGTH * REPEAT)

char* getToken(char* line, int num)
{
    char* token;
    for (token = strtok(line, "\t"); token != NULL; token = strtok(NULL, "\n")){
        if (!--num)
            return token;
    }
    return NULL;
}

void readData(const char* file, double* data){
    char* line = NULL;
    size_t len = 0;
    size_t read;

    FILE* fp = fopen(file, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    
    int temp = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        char* lineDup = strdup(line);
        char* token = getToken(lineDup, 2);
        data[temp] = atof(token);
        temp++;
         
        free(lineDup);
    }

    fclose(fp);
    if (line)
        free(line);
}

void writeToFile(const char* file, char** data, int dataLen){
    FILE* fp = fopen(file, "w");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    
    for (size_t i = 0; i < dataLen; i++)
    {
        fprintf(fp, "%s", data[i]);
    }

    fclose(fp);
}

void minAndMax(double* data, double* minMax){
    for(int i = 0; i < SIGNAL_LENGTH; ++i){
        if (data[i] > minMax[1])
            minMax[1] = data[i];
        if (data[i] < minMax[0])
            minMax[0] = data[i];
    }
}

void scaleData(double* data){
    double minMax[2];
    minAndMax(data, minMax);
    for(int i = 0; i < SIGNAL_LENGTH; ++i){
        data[i] = (data[i] - minMax[0]) / (minMax[1]-minMax[0]) * 2 - 1;
    }
}

void repeatData(double* repeatedData, double* data){
    for(int i = 0; i < REPEAT; ++i){
        for(int j = 0; j < SIGNAL_LENGTH; ++j){
            repeatedData[(i*SIGNAL_LENGTH)+j] = data[j];
        }
    }
}

int numOfCrossing(double* data, int dataLen){
    int isPositive = data[0] > 0;
    int numCross = 0;
    for(int i = 0; i < dataLen; ++i){
        if(data[i] > 0 && !isPositive){
            numCross++;
            isPositive = 1;
        }
        else if(data[i] < 0 && isPositive){
            numCross++;
            isPositive = 0;
        }
    }
    return numCross;
}

//////////////////////
////// MPI part //////
//////////////////////

int main(int argc, char** argv){
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double* data;
    double* repeatedData;
    if (rank == 0)
    {
        data = (double*) malloc(sizeof(double) * SIGNAL_LENGTH);
        readData("./ABPsignal.txt", data);
        scaleData(data);

        repeatedData = (double*) malloc(sizeof(double) * DATA_LEN);
        repeatData(repeatedData, data);
    }

    int result = 0;
    char* resultsStrArr[ITERATIONS];
    for (int i = 0; i < ITERATIONS; i++)
    {
        double startTime = clock();
        
        int localDataLen = DATA_LEN / size;
        double* localData = malloc(sizeof(double) * localDataLen);
 
        MPI_Scatter(
            repeatedData, localDataLen, MPI_DOUBLE, localData,
            localDataLen, MPI_DOUBLE, 0, MPI_COMM_WORLD
        );

        int localCrossings = numOfCrossing(localData, localDataLen);

        int* allCrossings = NULL;
        if (rank == 0) {
            allCrossings = malloc(sizeof(int) * size);
        }

        MPI_Gather(
            &localCrossings, 1, MPI_INT, allCrossings,
            1, MPI_INT, 0, MPI_COMM_WORLD
        );

        if(rank == 0){
            int totalCrossings = 0;

            for (int i = 0; i < size; i++) {
                totalCrossings += allCrossings[i];
            }
            
            printf("Num of x axis crossing: %d\n", totalCrossings);
            double timeDiff = clock() - startTime;

            double timeResult = (double)(timeDiff) / CLOCKS_PER_SEC;
            double timeResultInMilisec = timeResult * 1000;

            printf(
                "Time taken %.10f seconds and %f milliseconds (%d iterations)\n\n",
                timeResult, timeResultInMilisec, i
            );

            free(allCrossings);
            free(localData);
        }
    }

    MPI_Finalize();

    free(data);
    free(repeatedData);

    exit(EXIT_SUCCESS);
}
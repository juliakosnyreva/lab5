#include <mpi.h>
#include <iostream>
#include <chrono>
#include <random>

using namespace std;

int main(int argc, char** argv) {
    int rank, numtasks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    const int size = 4;
    int matrix[size][size + 1];

    //заполнение матрицы случайными числами
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(1, 1000);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size + 1; j++) {
            matrix[i][j] = dis(gen);
        }
    }
    //вывод элементов матрицы
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size + 1; j++) {
                cout << matrix[i][j] << "\t";
            }
            cout << endl;
        }
        cout << endl;
    }

    auto start_time = chrono::high_resolution_clock::now();

    //прямой ход метода Гаусса
    for (int iter = 0; iter < size; iter++) {
        MPI_Bcast(&matrix, (size + 1) * size, MPI_INT, 0, MPI_COMM_WORLD);
        for (int i = iter + 1 + rank; i < size; i += numtasks) {
            double mult = matrix[i][iter] / matrix[iter][iter];
            for (int j = iter; j <= size; j++) {
                matrix[i][j] -= mult * matrix[iter][j];
            }
        }

        if (rank != 0) {
            // Отправляем обновленную строку матрицы обратно процессу 0
            for (int i = iter + 1 + rank; i < size; i += numtasks) {
                MPI_Send(&matrix[i][0], size + 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Status status;
            for (int k = 1; k < numtasks; k++) {
                for (int i = iter + 1 + k; i < size; i += numtasks) {
                    MPI_Recv(&matrix[i][0], size + 1, MPI_INT, k, 0, MPI_COMM_WORLD, &status);
                }
            }
        }
    }
    //обратный ход
    if (rank == 0) {
        for (int i = size - 1; i >= 0; i--) {
            double sum = matrix[i][size];
            for (int j = i + 1; j < size; j++) {
                sum -= matrix[i][j] * matrix[j][size];
            }
            matrix[i][size] = sum / matrix[i][i];
            cout << "x" << i << " = " << matrix[i][size] << endl;
        }

        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        cout << "time: " << duration.count() << " microseconds" << endl;
    }

    MPI_Finalize();
    return 0;
}


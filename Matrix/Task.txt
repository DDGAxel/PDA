import java.util.Random;
import java.util.concurrent.*;

public class ExecutorServiceMatrixMultiplication {
	public static void main(String arg[]) throws Exception {
		final byte N = 5;
		int[][] dataA = new int[N][N];
		int[][] dataB = new int[N][N];

		Random rand = new Random();

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				dataA[i][j] = rand.nextInt(10);
				dataB[i][j] = rand.nextInt(10);
			}
		}

		Matrix.printMatrix(dataA);
		System.out.println();
		Matrix.printMatrix(dataB);
		System.out.println();

		ExecutorService executor = Executors.newFixedThreadPool(3);

		Matrix matrixA = new Matrix(dataA);
		Matrix matrixB = new Matrix(dataB);
		System.out.println(matrixA.multiply(matrixB));

		executor.shutdown();
	}

}
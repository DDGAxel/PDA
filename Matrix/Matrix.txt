import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Matrix {
	private static final int THREAD_COUNT = 3;

	private int rowCount;

	int colCount;

	int data[][];

	public Matrix(int data[][]) {
		this.data = data;
		this.rowCount = data.length;
		this.colCount = data[0].length;
	}

	public Matrix multiply(Matrix m) {
		ExecutorService executor = Executors.newFixedThreadPool(THREAD_COUNT);

		try {
			if (this.colCount != m.rowCount)
				throw new IllegalArgumentException("Matrix col count does not match argument's row count. ColCount="
						+ this.colCount + " RowCount=" + m.rowCount);

			List<Task> listFrag = Matrix.getTasks(this, m);
			List<Future<Integer>> listResult = executor.invokeAll(listFrag);

			int data[][] = new int[this.rowCount][m.colCount];
			for (int i = 0; i < this.rowCount; ++i)
				for (int j = 0; j < m.colCount; ++j)
					data[i][j] = listResult.get(i * m.colCount + j).get();

			return new Matrix(data);
		} catch (Exception e) {
			executor.shutdown();
		}
		return null;
	}

	public static List<Task> getTasks(Matrix matrixA, Matrix matrixB) {
		List<Task> listFrag = new ArrayList<>();
		for (int i = 0; i < matrixA.rowCount; ++i)
			for (int j = 0; j < matrixB.colCount; ++j)
				listFrag.add(new Task(matrixA, matrixB, i, j));
		return listFrag;
	}

	public String toString() {
		StringBuilder builder = new StringBuilder();
		for (int i = 0; i < rowCount; ++i) {
			for (int j = 0; j < colCount; ++j)
				builder.append(String.format(" %3d ", data[i][j]));
			builder.append('\n');
		}
		return builder.toString();
	}

	public static void printMatrix(int matrix[][]) {
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix.length; j++) {
				System.out.print(matrix[i][j] + " ");
			}
			System.out.println();
		}
	}

}
import java.util.concurrent.Callable;

public class Task implements Callable<Integer> {
	int rowId, colId;

	Matrix matrixA, matrixB;

	public Task(Matrix matrixA, Matrix matrixB, int rowId, int colId) {
		this.rowId = rowId;
		this.colId = colId;
		this.matrixA = matrixA;
		this.matrixB = matrixB;
	}

	@Override
	public Integer call() throws Exception {
		String mesg = String.format("A[%d] * B[%d]", rowId, colId);
		System.out.println("Started  Task. " + mesg);
		int product = 0;
		for (int i = 0; i < matrixA.colCount; ++i)
			product = product + matrixA.data[rowId][i] * matrixB.data[i][colId];
		System.out.println("Finished Task. " + mesg + " = " + product);
		return product;
	}

}
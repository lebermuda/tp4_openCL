__kernel void inverse_pass(__global double *data,
				  __global double *pivot, __global long *k, __global long *cols) {

	int idx = get_global_id(0);

	if (idx != *k) {
		double lValue = data[idx * *cols + *k];

		for (int j = 0; j < *cols; j++) {
			data[idx * *cols + j] -= pivot[j] * lValue;
		}
	}
}

    FILE *
OpenFile (
    const char * const fn_p,
    const char * const open_mode_p,
    const int if_silent            // If not show messages
    )
{
    FILE * f_p = NULL;

    if (fn_p == NULL) {
        printf ("Null file name pointer.");
        exit (-1);
    }

    if (!if_silent) {
        fprintf(stdout,"Opening the file %s ... ", fn_p);
    }

    f_p = fopen(fn_p, open_mode_p);
    if (f_p == NULL) {
        if (!if_silent) {
            fprintf(stdout,"failed.\n");
        } else {
            fprintf(stdout,"\nOpening the file %s ... failed.\n\n", fn_p);
        }
		exit (-1);
    }
    if (!if_silent) fprintf(stdout,"succeeded.\n");

    return (f_p);
}

    int
GenMatrixFile (
    const char * const matrix_fn_p,
    const unsigned int M_WIDTH,          // matrix width
    const unsigned int M_HEIGHT,         // matrix height
    const int if_silent         // If not show messages
    )
{
    FILE * matrix_fp = NULL;
    const unsigned int M_SIZE = M_WIDTH * M_HEIGHT;
    unsigned int * matrix = NULL;
    unsigned int i = 0, j = 0;

    matrix_fp = OpenFile (matrix_fn_p, "wb", 1);
    matrix = (unsigned int *) malloc (M_SIZE * sizeof (unsigned int));
    
    if (!if_silent) fprintf (stdout, "Generated contents of matrix:\n");
    for (i = 0; i < M_HEIGHT; i++) {
      for (j = 0; j < M_WIDTH; j++) {
        matrix[i*M_WIDTH + j] = i+j+1;
        //fwrite (&matrix[i*M_WIDTH + j], 1, sizeof (unsigned int), matrix_fp);
        if (!if_silent) fprintf (stdout, "%u ", matrix[i*M_WIDTH + j]);
      }
      if (!if_silent) fprintf (stdout, "\n");
    }
    fwrite (matrix, 1, M_SIZE * sizeof (unsigned int), matrix_fp);
    fclose (matrix_fp);
    free (matrix); matrix = NULL;

    return (1);
}

    double *
ReadDoubleMatrixFile (
    const char * const matrix_fn_p,
    const unsigned int M_WIDTH,          // matrix width
    const unsigned int M_HEIGHT,         // matrix height
    const int if_last,
    const int if_silent         // If not show messages
    )
{
    FILE * matrix_fp = NULL;
    const unsigned int M_SIZE = M_WIDTH * M_HEIGHT;
    double * matrix = NULL;
    unsigned int i = 0, j = 0;

    matrix_fp = OpenFile(matrix_fn_p, "rb", if_silent);
    matrix = (double *) malloc(M_SIZE * sizeof (double));
    if (if_last) 
        fseek(matrix_fp, (long)-1*M_SIZE*sizeof(double), SEEK_END);

    fread(matrix, 1, M_SIZE * sizeof (double), matrix_fp);

	if (!if_silent) {
	    fprintf (stdout, "Read contents of matrix:\n");
    	for (i = 0; i < M_HEIGHT; i++) {
    	    for (j = 0; j < M_WIDTH; j++) {
        	    fprintf (stdout, "%lf ", matrix[i*M_WIDTH + j]);
    	    }
    	    fprintf (stdout, "\n");
    	}
    }
    fclose (matrix_fp);

    return (matrix);
}

    float *
ReadMatrixFile (
    const char * const matrix_fn_p,
    const unsigned int M_WIDTH,          // matrix width
    const unsigned int M_HEIGHT,         // matrix height
    const int if_last,
    const int if_silent         // If not show messages
    )
{
    FILE * matrix_fp = NULL;
    const unsigned int M_SIZE = M_WIDTH * M_HEIGHT;
    float * matrix = NULL;
    unsigned int i = 0, j = 0;

    matrix_fp = OpenFile(matrix_fn_p, "rb", if_silent);
    matrix = (float *) malloc(M_SIZE * sizeof (float));
    if (if_last)
        fseek(matrix_fp, (long)-1*M_SIZE*sizeof(float), SEEK_END);
    fread(matrix, 1, M_SIZE * sizeof (float), matrix_fp);

	if (!if_silent) {
	    fprintf (stdout, "Read contents of matrix:\n");
    	for (i = 0; i < M_HEIGHT; i++) {
    	    for (j = 0; j < M_WIDTH; j++) {
        	    fprintf (stdout, "%f ", matrix[i*M_WIDTH + j]);
    	    }
    	    fprintf (stdout, "\n");
    	}
    }
    fclose (matrix_fp);

    return (matrix);
}

    int
WriteMatrixFile (
    const char * const matrix_fn_p,
    const float * const matrix,
    const unsigned int M_WIDTH,          // matrix width
    const unsigned int M_HEIGHT,         // matrix height
    const int if_silent         // If not show messages
    )
{
    FILE * matrix_fp = NULL;
    const unsigned int M_SIZE = M_WIDTH * M_HEIGHT;
    unsigned int i = 0, j = 0;

    matrix_fp = OpenFile (matrix_fn_p, "wb", if_silent);
    fwrite (matrix, 1, M_SIZE * sizeof (float), matrix_fp);

	if (!if_silent) {
	    fprintf (stdout, "Written contents of matrix:\n");
    	for (i = 0; i < M_HEIGHT; i++) {
    	    for (j = 0; j < M_WIDTH; j++) {
        	    fprintf (stdout, "%f ", matrix[i*M_WIDTH + j]);
    	    }
    	    fprintf (stdout, "\n");
    	}
    }
    fclose (matrix_fp);

    return (1);
}

    int
ContinueWriteMatrix(
    FILE * matrix_fp,
    const data_type * const matrix,
    const data_type tag,
    const unsigned int M_WIDTH,
    const unsigned int M_HEIGHT,
    const int if_silent
    )
{
    const unsigned int M_SIZE = M_WIDTH * M_HEIGHT;
    unsigned int i = 0, j = 0;

    fwrite (&tag, 1, sizeof(data_type), matrix_fp);
    fwrite (matrix, 1, M_SIZE * sizeof (data_type), matrix_fp);

	if (!if_silent) {
	    fprintf (stdout, "Written contents of matrix:\n");
    	for (i = 0; i < M_HEIGHT; i++) {
    	    for (j = 0; j < M_WIDTH; j++) {
        	    fprintf (stdout, "%f ", (float)matrix[i*M_WIDTH + j]);
    	    }
    	    fprintf (stdout, "\n");
    	}
    }
    return (1);

}

// Usage:
// CompareMatrixFile ("your output", "golden output", WC, HC, 1);

    void
CompareMatrixFile (
    const char * const matrix_fn_p1,
    const char * const matrix_fn_p2,
    const unsigned int M_WIDTH,          // matrix width
    const unsigned int M_HEIGHT,         // matrix height
    const int if_silent         // If not show messages
    )
{
    unsigned int i = 0, j = 0, wrong = 0;
    int check_ok = 1;
    float * m1 = ReadMatrixFile (matrix_fn_p1, M_WIDTH, M_HEIGHT, 0, if_silent);
    float * m2 = ReadMatrixFile (matrix_fn_p2, M_WIDTH, M_HEIGHT, 0, if_silent);

    printf ("  Comparing file %s with %s ...\n", matrix_fn_p1, matrix_fn_p2);
    for (i = 0; i < M_HEIGHT && wrong < 15; i++) {
        for (j = 0; j < M_WIDTH && wrong < 15; j++) {
                //printf ("m1[%d][%d] ?= m2[%d][%d] : %d ?= %d\n",
                //        i,j,i,j, m1[i*M_WIDTH+j], m2[i*M_WIDTH+j]);
            if (m1[i*M_WIDTH+j] != m2[i*M_WIDTH+j]) {
                printf ("m1[%d][%d] != m2[%d][%d] : %f != %f\n",
                        i,j,i,j, m1[i*M_WIDTH+j], m2[i*M_WIDTH+j]);
                check_ok = 0; wrong++;
            }
        }
    }
    printf ("    Check ok? ");
    if (check_ok) printf ("Passed.\n");
    else printf ("Failed.\n");
}

    float
CheckSum(const float *matrix, const int width, const int height)
{
    int i, j;
    float s1, s2;

    for (i = 0, s1 = 0; i < width; i++) {
        for (j = 0, s2 = 0; j < height; j++) {
            s2 += matrix[i * width + j];
        }
        s1 += s2;
    }

    return s1;
}



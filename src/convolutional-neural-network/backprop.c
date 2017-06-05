/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 *
 ******************************************************************
 */

#include <stdio.h>
#include <backprop.h>
#include <math.h>

#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))

#define fastcopy(to,from,len)\
{\
  register char *_to,*_from;\
  register int _i,_l;\
  _to = (char *)(to);\
  _from = (char *)(from);\
  _l = (len);\
  for (_i = 0; _i < _l; _i++) *_to++ = *_from++;\
}

/*** Return random number between 0.0 and 1.0 ***/
double drnd()
{
  return ((double) random() / (double) BIGRND);
}

/*** Return random number between -1.0 and 1.0 ***/
double dpn1()
{
  return ((drnd() * 2.0) - 1.0);
}

/*** The squashing function.  Currently, it's a sigmoid. ***/

double squash(x)
double x;
{
  return (1.0 / (1.0 + exp(-x)));
}


/*** Allocate 1d array of doubles ***/

double *alloc_1d_dbl(n)
int n;
{
  double *new;

  new = (double *) malloc ((unsigned) (n * sizeof (double)));
  if (new == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of doubles\n");
    return (NULL);
  }
  return (new);
}


/*** Allocate 2d array of doubles ***/

double **alloc_2d_dbl(m, n)
int m, n;
{
  int i;
  double **new;

  new = (double **) malloc ((unsigned) (m * sizeof (double *)));
  if (new == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (NULL);
  }

  for (i = 0; i < m; i++) {
    new[i] = alloc_1d_dbl(n);
  }

  return (new);
}


bpnn_randomize_weights(w, m, n)
double **w;
int m, n;
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = dpn1();
    }
  }
}


bpnn_zero_weights(w, m, n)
double **w;
int m, n;
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = 0.0;
    }
  }
}


void bpnn_initialize(seed)
{
  printf("Random number generator seed: %d\n", seed);
  srandom(seed);
}


BPNN *bpnn_internal_create(n_in, n_hidden, n_out)
int n_in, n_hidden, n_out;
{
  BPNN *newnet;

  newnet = (BPNN *) malloc (sizeof (BPNN));
  if (newnet == NULL) {
    printf("BPNN_CREATE: Couldn't allocate neural network\n");
    return (NULL);
  }

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  //newnet->convolution_n = n_in;

  newnet->input_units = alloc_1d_dbl(n_in + 1);
  newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
  newnet->output_units = alloc_1d_dbl(n_out + 1);

  //CONV
  newnet->convolution_units = alloc_1d_dbl(n_in + 1);
  newnet->kernel_units = alloc_2d_dbl(4, 4);

  newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
  newnet->output_delta = alloc_1d_dbl(n_out + 1);
  newnet->target = alloc_1d_dbl(n_out + 1);

  newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  return (newnet);
}


void bpnn_free(net)
BPNN *net;
{
  int n1, n2, i;

  n1 = net->input_n;
  n2 = net->hidden_n;

  free((char *) net->input_units);
  free((char *) net->hidden_units);
  free((char *) net->output_units);

  free((char *) net->kernel_units);
  free((char *) net->convolution_units);



  free((char *) net->hidden_delta);
  free((char *) net->output_delta);
  free((char *) net->target);

  for (i = 0; i <= n1; i++) {
    free((char *) net->input_weights[i]);
    free((char *) net->input_prev_weights[i]);
  }
  free((char *) net->input_weights);
  free((char *) net->input_prev_weights);

  for (i = 0; i <= n2; i++) {
    free((char *) net->hidden_weights[i]);
    free((char *) net->hidden_prev_weights[i]);
  }
  free((char *) net->hidden_weights);
  free((char *) net->hidden_prev_weights);

  free((char *) net);
}


/*** Creates a new fully-connected network from scratch,
     with the given numbers of input, hidden, and output units.
     Threshold units are automatically included.  All weights are
     randomly initialized.

     Space is also allocated for temporary storage (momentum weights,
     error computations, etc).
***/

BPNN *bpnn_create(n_in, n_hidden, n_out)
int n_in, n_hidden, n_out;
{
  double **A;

  BPNN *newnet;

  newnet = bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);

  A = newnet->kernel_units;
  A[0][0] = 0;
  A[0][1] = 0;
  A[0][2] = 0;

  A[1][0] = 0;
  A[1][1] = 1;
  A[1][2] = 0;

  A[2][0] = 0;
  A[2][1] = 0;
  A[2][2] = 0;

  //newnet->kernel_units = {{ 0, -1, 0 }, { -1, 5, -1 }, { 0, -1, 0}};

  return (newnet);
}



void bpnn_layerforward(l1, l2, conn, n1, n2)
double *l1, *l2, **conn;
int n1, n2;
{
  double sum;
  int j, k;

  /*** Set up thresholding unit ***/
  l1[0] = 1.0;
  l2[0] = 1.0;

  /*** For each unit in second layer ***/
  for (j = 1; j <= n2; j++) {

    /*** Compute weighted sum of its inputs ***/
    sum = 0.0;
    for (k = 0; k <= n1; k++) {
      sum += conn[k][j] * l1[k];
    }
    l2[j] = squash(sum);
  }

}
void bpnn_convolute(l1, l2, kern, n1, n2)
double *l1, *l2, **kern;
int n1, n2;
{
  double sum;
  int j, k, r, v;
  int l;
  l = 3;
  /*** Set up thresholding unit ***/
  l1[0] = 1.0;
  /*** For each unit in convolution layer ***/
  for (j = 1; j <= n1; j++) {
    /*** Compute weighted sum of its inputs ***/
    sum = 0.0;
    for (r = 0; r < l; r++) {
      for(k = 0; k < l; k++){
        if((j % 32 == 1 && k == 0 ) || (j % 32 == 0 && k == 2 )){ //add 1 padding to the left or right
          //do nothing
        }
        else if((j <= 32 && r == 0) || (j > 928 && r == 2)){ //add 1 padding at the top or the bottom
          //do nothing
        }
        else{
          sum += kern[r][k] * l1[ (k-1) + (j + (32 * (r-1)))];
        }
      }
    }
    l2[j] = (sum);
  }
}


void bpnn_output_error(delta, target, output, nj, err)
double *delta, *target, *output, *err;
int nj;
{
  int j;
  double o, t, errsum;

  errsum = 0.0;
  for (j = 1; j <= nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}


void bpnn_hidden_error(delta_h, nh, delta_o, no, who, hidden, err)
double *delta_h, *delta_o, *hidden, **who, *err;
int nh, no;
{
  int j, k;
  double h, sum, errsum;

  errsum = 0.0;
  for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j][k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}


void bpnn_adjust_weights(delta, ndelta, ly, nly, w, oldw, eta, momentum)
double *delta, *ly, **w, **oldw, eta, momentum;
{
  double new_dw;
  int k, j;

  ly[0] = 1.0;
  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      new_dw = ((eta * delta[j] * ly[k]) + (momentum * oldw[k][j]));
      w[k][j] += new_dw;
      oldw[k][j] = new_dw;
    }
  }
}


void bpnn_feedforward(net)
BPNN *net;
{
  int in, hid, out, conv;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  /*** Feed forward input activations. ***/
  printf("bpnn_convolute\n");
  /*for(int i = 0; i <= in; i++){
    printf("%f\n", net->input_units[i] );
  }*/

 /* bpnn_convolute(net->input_units, net->convolution_units,
      net->kernel_units, in, in);
  for(int i = 0; i <= in; i++){
    if(net->input_units[i] != net->convolution_units[i]){
        printf("%f\n", net->input_units[i] );

      printf("%f\n", net->convolution_units[i] );
    }
  }

  printf("trying to set input to convolution\n");
  net->input_units = net->convolution_units;
*/
  bpnn_layerforward(net->input_units, net->hidden_units,
      net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
      net->hidden_weights, hid, out);

}


void bpnn_train(net, eta, momentum, eo, eh)
BPNN *net;
double eta, momentum, *eo, *eh;
{
  int in, hid, out, conv;
  double out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  /*** Feed forward input activations. ***/
  bpnn_convolute(net->input_units, net->convolution_units,
      net->kernel_units, in, conv);

  net->input_units = net->convolution_units;
  
  bpnn_layerforward(net->input_units, net->hidden_units,
      net->input_weights, in, hid);
  bpnn_layerforward(net->input_units, net->hidden_units,
      net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
      net->hidden_weights, hid, out);

  /*** Compute error on output and hidden units. ***/
  bpnn_output_error(net->output_delta, net->target, net->output_units,
      out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
      net->hidden_weights, net->hidden_units, &hid_err);
  *eo = out_err;
  *eh = hid_err;

  /*** Adjust input and hidden weights. ***/
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
      net->hidden_weights, net->hidden_prev_weights, eta, momentum);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
      net->input_weights, net->input_prev_weights, eta, momentum);

}




void bpnn_save(net, filename)
BPNN *net;
char *filename;
{
  int fd, n1, n2, n3, i, j, memcnt;
  double dvalue, **w;
  char *mem;

  if ((fd = creat(filename, 0644)) == -1) {
    printf("BPNN_SAVE: Cannot create '%s'\n", filename);
    return;
  }

  n1 = net->input_n;  n2 = net->hidden_n;  n3 = net->output_n;
  printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
  fflush(stdout);

  write(fd, (char *) &n1, sizeof(int));
  write(fd, (char *) &n2, sizeof(int));
  write(fd, (char *) &n3, sizeof(int));

  memcnt = 0;
  w = net->input_weights;
  mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(double)));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      dvalue = w[i][j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(double));
      memcnt += sizeof(double);
    }
  }
  write(fd, mem, (n1+1) * (n2+1) * sizeof(double));
  free(mem);

  memcnt = 0;
  w = net->hidden_weights;
  mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(double)));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      dvalue = w[i][j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(double));
      memcnt += sizeof(double);
    }
  }
  write(fd, mem, (n2+1) * (n3+1) * sizeof(double));
  free(mem);

  close(fd);
  return;
}


BPNN *bpnn_read(filename)
char *filename;
{
  char *mem;
  BPNN *new;
  int fd, n1, n2, n3, i, j, memcnt;

  if ((fd = open(filename, 0, 0644)) == -1) {
    return (NULL);
  }

  printf("Reading '%s'\n", filename);  fflush(stdout);

  read(fd, (char *) &n1, sizeof(int));
  read(fd, (char *) &n2, sizeof(int));
  read(fd, (char *) &n3, sizeof(int));

  new = bpnn_internal_create(n1, n2, n3);

  printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
  printf("Reading input weights...");  fflush(stdout);

  memcnt = 0;
  mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(double)));
  read(fd, mem, (n1+1) * (n2+1) * sizeof(double));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      fastcopy(&(new->input_weights[i][j]), &mem[memcnt], sizeof(double));
      memcnt += sizeof(double);
    }
  }
  free(mem);

  printf("Done\nReading hidden weights...");  fflush(stdout);

  memcnt = 0;
  mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(double)));
  read(fd, mem, (n2+1) * (n3+1) * sizeof(double));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      fastcopy(&(new->hidden_weights[i][j]), &mem[memcnt], sizeof(double));
      memcnt += sizeof(double);
    }
  }
  free(mem);
  close(fd);

  printf("Done\n");  fflush(stdout);

  bpnn_zero_weights(new->input_prev_weights, n1, n2);
  bpnn_zero_weights(new->hidden_prev_weights, n2, n3);

  return (new);
}

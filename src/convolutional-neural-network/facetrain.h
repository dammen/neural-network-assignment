
#include "pgmimage.h"
#include "backprop.h"
#include "imagenet.h"

extern char *strcpy();
extern void exit();
extern int atoi(char argv[]);
 

void backprop_face(IMAGELIST *trainlist, IMAGELIST *test1list, IMAGELIST *test2list, int epochs, int savedelta, char *netname, int list_errors);
void printusage(char *prog);
void performance_on_imagelist(BPNN *net, IMAGELIST *il, int list_errors);
int evaluate_performance(BPNN *net, double *err);
int output_result_on_imagelist(BPNN *net, IMAGELIST *il, int list_errors);

char *names[4];
#include <pgmimage.h>
#include <backprop.h>
extern void exit();


void load_target(IMAGE *img, BPNN *net);
void load_input_with_image(IMAGE *img, BPNN *net);
char *names[4];
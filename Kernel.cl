
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void conv4x8(__global float* images, int W, int H)
{
    for (int i = 0; i < H*CHANNELNUM; ++i)
    {
        for (int j = 0; j < W; ++j)
        {
            printf("%f ", images[i*W+j]);
        }
        printf("\n");
    }
}

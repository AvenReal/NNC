
#include "ActivationFunctions.h"
#include "MainHeader.h"
#include "layer.h"

/**
 * Activation function used inside the neural network
 * @param layer The layer.
 */
void ReLU_ActivationFunction(const struct layer *layer)
{
    for (int i = 0; i < layer->layer_size; i++)
    {
        layer->outputs[i] = layer->outputs[i] > 0 ? layer->outputs[i] : 0;
    }
}

/**
 * Activation function used for the last layer (output) of the neural network
 * @param layer The layer
 */
void SoftMax_ActivationFunction(const struct layer *layer)
{
    long double sum = 0;
    long double max = layer->outputs[0];
    for (int i = 1; i < layer->layer_size; i++)
    {
        if (layer->outputs[i] > max)
        {
            max = layer->outputs[i];
        }
    }

    for (int i = 0; i < layer->layer_size; i++)
    {
        layer->outputs[i] = exp(layer->outputs[i] - max);
        sum += layer->outputs[i];
    }

    for (int i = 0; i < layer->layer_size; i++)
    {
        layer->outputs[i] = layer->outputs[i] / sum;
        if (isnanf(layer->outputs[i]))
        {
            errx(EXIT_FAILURE,
            "output is nan | layer->outputs[i] / sum : %Lg / %Lg",
                 layer->outputs[i], sum);
        }
    }
}
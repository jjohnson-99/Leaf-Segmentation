import numpy as np


def rl_decode(enc, HEIGHT, WIDTH):
    """Decode a run-length encoded array.
    
    Args:
        enc: String represenation of a run-length encoded HEIGHT x WIDTH array.
    
    Returns:
        A HEIGHT x WIDTH decoded numpy array.

    Example:
        >>> enc = "3 0 2 1 1 0 2 1 2 0 4 1 1 0"
        >>> rl_decode(example, HEIGHT=3, WIDTH=5)
        array([[0, 0, 0, 1, 1],
               [0, 1, 1, 0, 0],
               [1, 1, 1, 1, 0]], dtype=uint8)
        >>> enc = "3 3 2 1 1 0 2 1 2 0 4 1 1 0"
        >>> rl_decode(example, HEIGHT=3, WIDTH=5)
        array([[3, 3, 3, 1, 1],
               [0, 1, 1, 0, 0],
               [1, 1, 1, 1, 0]], dtype=uint8)
    """
    parts = [int(s) for s in enc.split(' ')]
    dec = list()
    for i in range(0, len(parts), 2):
        cnt = parts[i]
        val = parts[i + 1]
        dec += cnt * [val]
    return np.array(dec, dtype=np.uint8).reshape((HEIGHT, WIDTH))


def rl_encode(mask):
    """Run-length encode an array.

    Args:
        mask: An arbitrary array to be encoded.
    
    Returns:
        A string representing the run-length encoding of mask.

    Example:
        >>> mask = array([[0, 0, 0, 1, 1],
                          [0, 1, 1, 0, 0],
                          [1, 1, 1, 1, 0]], dtype=uint8)
        >>> rl_encode(mask)
        "3 0 2 1 1 0 2 1 2 0 4 1 1 0"
    """
    pixels = mask.flatten()
    rle = []
    last_value = pixels[0]
    count = 1
    for i, pixel in enumerate(pixels[1:]):
        if pixel != last_value:
            rle.append(count)
            rle.append(last_value)

            last_value = pixel
            count = 1
        else:
            count += 1

    rle.append(count)
    rle.append(last_value)

    return ' '.join(str(x) for x in rle)

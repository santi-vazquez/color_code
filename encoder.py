from PIL import Image, ImageDraw
import random

#00 = white, 01 = black, 10 = red, 11 = green

def string_to_bit(message: str):
    code = []
    for char in message:
        num = ord(char)
        bin_num = f"{num:08b}"
        for i in range(0, 8, 2):
            code.append(bin_num[i:i+2])
    return code

def draw_color_code(size = int, box_size = int, message = str):
    """
    size = number of boxes along one side of the QR grid (>18)
    box_size = the pixel size of each box 
    message = message you want to encode

    outputs an image of color_code. Black top right corner, green top left corner, 
    red bottom right corner. The message encoded to stop decoding is '@$?STOPplease'.
    The rest of the boxes are random.
    """
    bit_to_color = {
        '00': (255, 255, 255),
        '01': (0, 0, 0),
        '10': (255, 0, 0),
        '11': (0, 255, 0),
    }

    # Calculate the total image size
    img_size = size * box_size

    # Create a new image with a white background
    img = Image.new('RGB', (img_size, img_size), 'white')  # '1' for 1-bit pixels, black and white
    draw = ImageDraw.Draw(img)

    #draw indicator rectangles
    top_left = (0,0)
    x0, y0 = [x * box_size for x in top_left]
    x1, y1 = [x0 + 7 * box_size, y0 + 7 * box_size]
    draw.rectangle([x0, y0, x1 + box_size, y1 + box_size], fill='white')
    draw.rectangle([x0, y0, x1, y1], fill=0)  # black square
    draw.rectangle([x0 + box_size, y0 + box_size, x1 - box_size, y1 - box_size], fill='white')  
    draw.rectangle([x0 + 2 * box_size, y0 + 2 * box_size, x1 - 2 * box_size, y1 - 2 * box_size], fill=0) 

    bottom_left = (0, size - 7)
    x0, y0 = [x * box_size for x in bottom_left]
    x1, y1 = [x0 + 7 * box_size, y0 + 7 * box_size]
    draw.rectangle([x0, y0 - box_size, x1 + box_size, y1 + box_size], fill='white')
    draw.rectangle([x0, y0, x1, y1], fill=(255,0,0))  # black square
    draw.rectangle([x0 + box_size, y0 + box_size, x1 - box_size, y1 - box_size], fill='white')  
    draw.rectangle([x0 + 2 * box_size, y0 + 2 * box_size, x1 - 2 * box_size, y1 - 2 * box_size], fill=(255,0,0))

    top_right = (size - 7, 0)
    x0, y0 = [x * box_size for x in top_right]
    x1, y1 = [x0 + 7 * box_size, y0 + 7 * box_size]
    draw.rectangle([x0 - box_size, y0, x1 + box_size, y1 + box_size], fill='white')
    draw.rectangle([x0, y0, x1, y1], fill=(0,255,0))  # black square
    draw.rectangle([x0 + box_size, y0 + box_size, x1 - box_size, y1 - box_size], fill='white')  
    draw.rectangle([x0 + 2 * box_size, y0 + 2 * box_size, x1 - 2 * box_size, y1 - 2 * box_size], fill=(0,255,0))

    #get coordinates of all possible box locations in order
    order = []
    
    for i in range(size):
        for j in range(size):
            if (i < 8 and j < 8) or (i > (size - 9) and j < 8) or (i < 8 and j > (size - 9)):
                pass
            else:
                x0 = j * box_size
                y0 = i * box_size
                x1 = x0 + box_size
                y1 = y0 + box_size
                order.append((x0, y0, x1, y1))

    #draw on message from left to right, top to bottom
    stop_message = '@$?STOPplease'
    qr_message = message + stop_message
    code = string_to_bit(qr_message)

    index = 0
    for bit in code:
        draw.rectangle(order[index], fill=bit_to_color[bit])
        index += 1

    #fill in remaining pixels with random colors
    while (index < len(order)):
        random_color = random.choice(list(bit_to_color.values()))
        draw.rectangle(order[index], fill=random_color)
        index += 1

    return img

# Example usage:
img = draw_color_code(29, 10, 'https://mec.mit.edu/')  # 29x29 grid, 10 pixels per box
img.show()

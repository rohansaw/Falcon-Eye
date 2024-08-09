def slice_image(image, slice_size, overlap_factor):
    slices = []
    start_points = []
    overlap = [
        int(slice_size[0] * overlap_factor[0]),
        int(slice_size[1] * overlap_factor[1]),
    ]
    for x in range(0, image.shape[1], slice_size[0] - overlap[0]):
        for y in range(0, image.shape[2], slice_size[1] - overlap[1]):
            start_points.append((x, y))
            slices.append(image[:, x : x + slice_size[0], y : y + slice_size[1]])
    return slices, start_points


def nms(segmentation_grid, iou_threshold):
    raise Exception("Not implemented")

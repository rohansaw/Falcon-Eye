from metrics import Metrics

# wrapper for coco that converts to the correct format of batch

class GridMetricsCOCO(Metrics):

    def process_batch(self, logits_grid_batch, y_batch):
        for logits_grid, y in zip(logits_grid_batch, y_batch):
            logits_grid = logits_grid.cpu().detach()
            self.tp_fp_tn_fn(logits_grid, y)
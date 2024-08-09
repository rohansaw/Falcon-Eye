from pytorch_lightning.callbacks import Callback

class ValidateBeforeTraining(Callback):
     def on_train_start(self, trainer, pl_module):
        return trainer.run_evaluation(test_mode=False)
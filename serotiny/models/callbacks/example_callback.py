#!/usr/bin/env python3

import pytorch_lightning as pl


class MyPrintingCallback(pl.Callback):
    def on_init_start(self, trainer):
        print("Starting to init trainer!")

    def on_init_end(self, trainer):
        print("Trainer is init now")

    def on_train_end(self, trainer, pl_module):
        print("Do something when training ends")

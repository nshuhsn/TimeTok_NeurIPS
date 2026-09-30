from mup.layer import MuReadout


class MuReadoutFSDP(MuReadout):
    """Version of mup's MuReadout with FSDP fix."""

    def width_mult(self):
        if hasattr(self.weight, "infshape"):
            width_mult = self.weight.infshape.width_mult()
        elif hasattr(self, "weight_infshape"):
            width_mult = self.weight_infshape.width_mult()
        else:
            raise AssertionError(
                "Please call set_base_shapes(...). If using torch.nn.DataParallel, "
                "switch to distributed training with "
                "torch.nn.parallel.DistributedDataParallel instead"
            )
        return width_mult

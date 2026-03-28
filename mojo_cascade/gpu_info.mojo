from std.gpu.host import DeviceContext

def main() raises:
    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    print("Compute capability:", ctx.compute_capability())


import os
import site
import glob

def find_cuda_paths():
    # Get site-packages path
    sp = site.getsitepackages()[0]

    # Search for cudnn and cuda_runtime DLL directories
    paths = []
    cudnn_lib = os.path.join(sp, 'nvidia', 'cudnn', 'lib')
    cuda_lib = os.path.join(sp, 'nvidia', 'cuda_runtime', 'lib')

    if os.path.isdir(cudnn_lib) and glob.glob(os.path.join(cudnn_lib, '*.dll')):
        paths.append(cudnn_lib)
    else:
        print("❌ cuDNN DLLs not found in:", cudnn_lib)

    if os.path.isdir(cuda_lib) and glob.glob(os.path.join(cuda_lib, '*.dll')):
        paths.append(cuda_lib)
    else:
        print("❌ CUDA Runtime DLLs not found in:", cuda_lib)

    return paths, sp

def write_pth(paths, site_packages_path):
    if not paths:
        print("❌ No valid CUDA/cuDNN DLL paths found. Aborting.")
        return

    pth_path = os.path.join(site_packages_path, 'cuda_dlls.pth')

    with open(pth_path, 'w') as f:
        for p in paths:
            f.write(p + '\n')

    print(f"✅ Wrote .pth file to: {pth_path}")
    for p in paths:
        print("  →", p)

if __name__ == "__main__":
    paths, sp = find_cuda_paths()
    write_pth(paths, sp)

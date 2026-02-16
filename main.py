from src.plotting import (
        plot_image_and_segmentation,
)

def main():
    print("Hello from leaf-segmentation!")
    plot_image_and_segmentation('leaf01', "./datasets")


if __name__ == "__main__":
    main()

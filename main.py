import quad_gen.get_models as get_models


def main():
    print("starting conversion")
    get_models.save_result(
        model_dir="/Users/devesh/Desktop/Coding/research/college/sim2multireal/quad_sim2multireal/input_model",
        out_dir="/Users/devesh/Desktop/Coding/research/college/sim2multireal/quad_sim2multireal/output_model",
    )
    print("done")


if __name__ == "__main__":
    main()
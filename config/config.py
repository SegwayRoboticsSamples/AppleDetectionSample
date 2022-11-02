import yaml
import argparse

def visualize_config(conf):
    """
    Visualize the configuration on the terminal to check the state
    :param conf:
    :return:
    """
    print("\nUsing this arguments check it\n")
    for key, value in sorted(conf.items()):
        if value is not None:
            print("{} -- {} --".format(key, value))
    print("\n\n")


def parse_confg():
    """
    Parse Configuration from YAML file
    :return args:
    """
    parser = argparse.ArgumentParser(description="detection using tensorflow")
    parser.add_argument('--load_config', default='/raid/yanx/Project/yolov3/experiments/cfg/yolov3_mobilenet_v1_test.yaml', #None,
                        dest='config_path')
    args, _ = parser.parse_known_args()
    if args.config_path:
        with open(args.config_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            return conf

def get_params():
    print("\nParsing Arguments..\n")
    conf = parse_confg()
    visualize_config(conf)
    #input()
    return conf

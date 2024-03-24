import yaml
from ast import literal_eval


class Config:
    def __init__(self, config):
        # Use object.__setattr__ as we are shadowing
        # __setattr__ by overriding it
        object.__setattr__(self, "_data", {})
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool, list, tuple)):
                if isinstance(value, str):
                    try:
                        v = literal_eval(value)
                        if isinstance(v, (list, tuple, bool)):
                            value = v
                        else:
                            raise ValueError
                    except (SyntaxError, ValueError):
                        try:
                            value = int(value)
                        except ValueError:
                            try:
                                value = float(value)
                            except ValueError:
                                pass  # stick to str
                self._data[key] = value
            elif isinstance(value, dict):
                self._data[key] = Config(value)
            else:
                raise ValueError(
                    f"Only strings, numbers, booleans, and dicts allowed. No {type(value)} for {key}."
                )

    # new
    def __getitem__(self, index):
        return self._data.get(index, None)

    def __getattr__(self, key):
        return self._data.get(key, None)

    def __setattr__(self, key, value):
        if key == "_data":
            object.__setattr__(self, key, value)
        else:
            self._data[key] = value

    def __setitem__(self, index, value):
        def create_dict_from_path(path, value):
            # leaf
            if not path:
                return value

            keys = path.split(".")

            return {keys[0]: create_dict_from_path(".".join(keys[1:]), value)}

        d = create_dict_from_path(index, value)
        key = index.split(".")[0]
        # Create new config based on dict and potentially merge existing data
        self._data[key] = Config(self._data[key].todict() | d[key])

    # New
    def __iter__(self):
        for element in self._data:
            yield self.__getattr__(element)

    def todict(self):
        d = {}
        for key, value in self._data.items():
            if isinstance(value, Config):
                d[key] = value.todict()
            else:
                d[key] = value
        return d

    def __repr__(self):
        return yaml.dump(self.todict())

    @classmethod
    def from_yaml_str(cls, str):
        parsed_dict = yaml.safe_load(str)
        return cls(parsed_dict)

    @classmethod
    def from_yaml_file(cls, path):
        with open(path, "r") as f:
            parsed_dict = yaml.safe_load(f.read())
        return cls(parsed_dict)

import json
import re


def json_prettyprint(obj) -> str:
    """
    Return a serialized dictionary as pretty-printed JSON. Lists of numbers are formatted using one line.
    """
    serialized_data = json.dumps(obj, indent=2)

    # Regex pattern to remove new lines between "[" and "]"
    pattern = re.compile(r'\[\s*([^"]*?)\s*\]', re.DOTALL)
    serialized_data = pattern.sub(lambda m: "[" + " ".join(m.group(1).split()) + "]", serialized_data)
    return serialized_data

from .api import app
from .clockwork import GuardianThreadMatrix

_guardian = GuardianThreadMatrix()
_guardian.bootstrap()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

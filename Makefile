# Contributing

## Setup

`ash
git clone https://github.com/faraz334/ImageEcho_v2.git
cd ImageEcho_v2
python -m venv venv
venv\Scripts\activate     # Windows
pip install -r requirements.txt
`

## Adding a New Engine

1. Create imageecho/engines/my_engine.py
2. Inherit from BaseEngine
3. Set 
ame = "my_engine"
4. Implement _perturb(self, x, target_class) -> torch.Tensor
5. Register in imageecho/engines/__init__.py
6. Add tests in 	ests/test_engines.py

`python
class MyEngine(BaseEngine):
    name = "my_engine"

    def _perturb(self, x, target_class=None):
        grad = self.surrogate.get_gradients(x)
        return x + self.epsilon * grad.sign()
`

## Running Tests

`ash
pytest tests/ -v
`

## Code Style

`ash
black imageecho/ gui/ tests/
flake8 imageecho/ --max-line-length=100
`
"@ | Out-File -FilePath docs/CONTRIBUTING.md -Encoding utf8
@"
.PHONY: run test lint clean

run:
python main.py

test:
pytest tests/ -v --tb=short

lint:
flake8 imageecho/ gui/ --max-line-length=100 --ignore=E501,W503

clean:
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete
rm -f adversarial*.png adv_*.png benchmark_report.md

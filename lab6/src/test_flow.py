from metaflow import FlowSpec, step

class TestFlow(FlowSpec):

    @step
    def start(self):
        print("🚀 Metaflow started!")
        self.next(self.end)

    @step
    def end(self):
        print("✅ Metaflow ended!")

if __name__ == '__main__':
    TestFlow()
import {app} from "../../scripts/app.js";
import {ComfyWidgets} from "../../scripts/widgets.js";
import {createTextWidget} from "./utils.js"

app.registerExtension({
    name: "sd_prompt_reader.extractorDisplay",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SDParameterExtractor") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                // Create widgets
                const styles = {textAlign: "center", fontSize: "0.75rem"}
                const value_display =  createTextWidget(app, this, "value_display", styles);
            };

            // Update widgets
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                this.widgets.find(obj => obj.name === "value_display").value = message.text[1]
                this.widgets.find(obj => obj.name === "parameter").options.values = message.text[0]
                if (this.widgets.find(obj => obj.name === "parameter").value === "parameters not loaded") {
                    this.widgets.find(obj => obj.name === "parameter").value = message.text[0][0]
                }
            };
        }
    },
});
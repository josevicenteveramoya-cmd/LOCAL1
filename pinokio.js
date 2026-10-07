const path = require('path')

module.exports = {
  title: "Real-ESRGAN Local",
  description: "Escalador de imágenes ligero con Real-ESRGAN",
  icon: "icon.png",
  menu: async (kernel) => {
    let installed = await kernel.exists(__dirname, "app", "env")
    let running = await kernel.running(__dirname, "start.json")
    if (installed) {
      if (running) {
        let local = kernel.memory.local[path.resolve(__dirname, "start.json")]
        return [{
          icon: "fa-solid fa-rocket",
          text: "Open Web UI",
          href: (local && local.url) ? local.url : "http://127.0.0.1:7860"
        }, {
          icon: "fa-solid fa-power-off",
          text: "Stop",
          href: "start.json",
          action: "stop"
        }]
      } else {
        return [{
          icon: "fa-solid fa-play",
          text: "Start",
          href: "start.json"
        }]
      }
    } else {
      return [{
        icon: "fa-solid fa-download",
        text: "Install",
        href: "install.json"
      }]
    }
  }
}
<template>
  <ImageCropLabeller
    :srcUrl="this.imageUrl"
    :imageId="this.imageId"
    :autoIncrement="true"
    :regionNames="this.regionNames"
  />
</template>

<script>
import ImageCropLabeller from './components/ImageCropLabeller.vue'

export default {
  name: 'App',
  components: {
    ImageCropLabeller
  },
  data () {
    return {
      imageId: '',
      imageUrl: '',
      regionNames: [
        'Main Page Top Left',
        'Main Page Top Right',
        'Main Page Bottom Left',
        'Main Page Bottom Right',
        'Upper Swatches Top Left',
        'Upper Swatches Top Right',
        'Upper Swatches Bottom Left',
        'Lower Swatches Top Right',
        'Lower Swatches Bottom Left',
        'Lower Swatches Bottom Right'
      ]
    }
  },
  created () {
    this.fetchData()
  },
  methods: {
    async fetchData () {
      const targetAnnotations = encodeURIComponent(this.regionNames.join(','))
      console.log(targetAnnotations)
      const url =
        'https://cianh-anna-atkins.azurewebsites.net/api/images/random-unannotated?code=yF1OvJvYyw7qTS0qZuGhoOShtO5mtWKzPfEs34aIHPKwXYDaE6eu2A==&missing-any-or-all=any&target-annotations=' +
        targetAnnotations
      const response = await (await fetch(url)).json()
      console.log(response)
      this.imageId = response.image_id
      this.imageUrl = response.url
    }
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
</style>
